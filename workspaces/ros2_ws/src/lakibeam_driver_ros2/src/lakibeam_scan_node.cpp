#include "lakibeam_driver_ros2/data_type.h"
#include "lakibeam_driver_ros2/remote.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr std::uint16_t kDataFlag = 0xEEFF;
constexpr double kDegToRad = M_PI / 180.0;

std::string bool_to_rest(bool value) { return value ? "true" : "false"; }
}  // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("lakibeam_scan_node");

  node->declare_parameter<std::string>("frame_id", "laser");
  node->declare_parameter<std::string>("output_topic", "/scan");
  node->declare_parameter<bool>("inverted", false);
  node->declare_parameter<std::string>("hostip", "0.0.0.0");
  node->declare_parameter<std::string>("sensorip", "192.168.8.2");
  node->declare_parameter<int>("port", 2368);
  node->declare_parameter<int>("angle_offset", 0);
  node->declare_parameter<int>("scanfreq", 30);
  node->declare_parameter<int>("filter", 3);
  node->declare_parameter<bool>("laser_enable", true);
  node->declare_parameter<int>("scan_range_start", 90);
  node->declare_parameter<int>("scan_range_stop", 270);

  const auto frame_id = node->get_parameter("frame_id").as_string();
  const auto output_topic = node->get_parameter("output_topic").as_string();
  const auto inverted = node->get_parameter("inverted").as_bool();
  const auto host_ip = node->get_parameter("hostip").as_string();
  const auto sensor_ip = node->get_parameter("sensorip").as_string();
  const auto port = node->get_parameter("port").as_int();
  const auto angle_offset = node->get_parameter("angle_offset").as_int();
  const auto scanfreq = node->get_parameter("scanfreq").as_int();
  const auto filter = node->get_parameter("filter").as_int();
  const auto laser_enable = node->get_parameter("laser_enable").as_bool();
  const auto scan_range_start = node->get_parameter("scan_range_start").as_int();
  const auto scan_range_stop = node->get_parameter("scan_range_stop").as_int();
  const auto scan_span_deg = std::max(0, static_cast<int>(scan_range_stop - scan_range_start));
  const auto centered_min_deg = -0.5 * static_cast<double>(scan_span_deg) + static_cast<double>(angle_offset);
  const auto centered_max_deg = 0.5 * static_cast<double>(scan_span_deg) + static_cast<double>(angle_offset);
  const auto logger = node->get_logger();

  auto scan_pub = node->create_publisher<sensor_msgs::msg::LaserScan>(output_topic, rclcpp::SensorDataQoS());

  RCLCPP_INFO(logger, "Lakibeam ROS2 驱动启动: sensor=%s host=%s:%ld topic=%s frame=%s",
              sensor_ip.c_str(), host_ip.c_str(), port, output_topic.c_str(), frame_id.c_str());

  sensor_config(sensor_ip, "/api/v1/sensor/scanfreq", std::to_string(scanfreq), logger);
  sensor_config(sensor_ip, "/api/v1/sensor/laser_enable", bool_to_rest(laser_enable), logger);
  sensor_config(sensor_ip, "/api/v1/sensor/scan_range/start", std::to_string(scan_range_start), logger);
  sensor_config(sensor_ip, "/api/v1/sensor/scan_range/stop", std::to_string(scan_range_stop), logger);
  sensor_config(sensor_ip, "/api/v1/sensor/filter", std::to_string(filter), logger);
  rclcpp::sleep_for(std::chrono::seconds(2));
  get_telemetry_data(sensor_ip, logger);

  const int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    RCLCPP_ERROR(logger, "创建 UDP socket 失败");
    rclcpp::shutdown();
    return 1;
  }

  int reuse = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in server_addr {};
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = inet_addr(host_ip.c_str());
  server_addr.sin_port = htons(static_cast<uint16_t>(port));

  if (bind(sockfd, reinterpret_cast<sockaddr *>(&server_addr), sizeof(server_addr)) < 0) {
    RCLCPP_ERROR(logger, "UDP bind 失败: host=%s port=%ld", host_ip.c_str(), port);
    close(sockfd);
    rclcpp::shutdown();
    return 1;
  }

  std::vector<ScanResponse> scan_vec;
  scan_vec.reserve(2048);
  MsopData msop_data {};
  rclcpp::Time scan_begin = node->get_clock()->now();
  rclcpp::Time scan_end = scan_begin;
  bool scan_vec_ready = false;
  int resolution = 25;
  int block_index = 12;
  std::size_t publish_count = 0;

  while (rclcpp::ok()) {
    if (!scan_vec_ready) {
      while (rclcpp::ok()) {
        if (block_index == 12) {
          sockaddr_in client_addr {};
          socklen_t client_len = sizeof(client_addr);
          const auto received = recvfrom(
              sockfd,
              &msop_data,
              sizeof(msop_data),
              0,
              reinterpret_cast<sockaddr *>(&client_addr),
              &client_len);
          if (received <= 0) {
            continue;
          }

          if (msop_data.blocks[0].azimuth == 0) {
            scan_end = scan_begin;
            scan_begin = node->get_clock()->now();
          }
          if (msop_data.blocks[1].azimuth > msop_data.blocks[0].azimuth) {
            resolution = std::max<int>((msop_data.blocks[1].azimuth - msop_data.blocks[0].azimuth) / 16, 1);
          }
          block_index = 0;
        }

        for (; block_index < 12; ++block_index) {
          for (int point_index = 0; point_index < 16; ++point_index) {
            ScanResponse response {};
            response.angle = msop_data.blocks[block_index].azimuth + resolution * point_index;
            if (msop_data.blocks[block_index].data_flag == kDataFlag) {
              if (response.angle == 0 && !scan_vec.empty() && !scan_vec_ready) {
                scan_vec_ready = true;
                if (scan_vec.size() < 1200) {
                  block_index = 12;
                }
                break;
              }
              response.dist = msop_data.blocks[block_index].result[point_index].dist_1;
              response.rssi = msop_data.blocks[block_index].result[point_index].rssi_1;
              scan_vec.push_back(response);
            }
          }
          if (scan_vec_ready) {
            break;
          }
        }
        if (scan_vec_ready) {
          break;
        }
      }
    }

    if (scan_vec_ready && !scan_vec.empty()) {
      sensor_msgs::msg::LaserScan scan_msg;
      const auto num_readings = static_cast<std::size_t>(scan_vec.size());
      double duration = (scan_begin - scan_end).seconds();
      if (duration <= 0.0) {
        duration = 1.0 / std::max<int>(scanfreq, 1);
      }

      scan_msg.header.stamp = scan_begin;
      scan_msg.header.frame_id = frame_id;
      scan_msg.angle_min = centered_min_deg * kDegToRad;
      scan_msg.angle_max = centered_max_deg * kDegToRad;
      if (num_readings > 1) {
        scan_msg.angle_increment = (scan_msg.angle_max - scan_msg.angle_min) / static_cast<double>(num_readings - 1);
      } else {
        scan_msg.angle_increment = 0.0;
      }
      scan_msg.scan_time = duration;
      scan_msg.time_increment = duration / static_cast<double>(num_readings);
      scan_msg.range_min = 0.0;
      scan_msg.range_max = 100.0;
      scan_msg.ranges.resize(num_readings);
      scan_msg.intensities.resize(num_readings);

      for (std::size_t index = 0; index < num_readings; ++index) {
        const auto range_m = static_cast<float>(scan_vec[index].dist) / 1000.0f;
        const auto intensity = static_cast<float>(scan_vec[index].rssi);
        if (!inverted) {
          scan_msg.ranges[index] = range_m;
          scan_msg.intensities[index] = intensity;
        } else {
          const auto mirrored_index = num_readings - index - 1;
          scan_msg.ranges[mirrored_index] = range_m;
          scan_msg.intensities[mirrored_index] = intensity;
        }
      }

      scan_pub->publish(scan_msg);
      ++publish_count;
      if (publish_count == 1 || publish_count % 30 == 0) {
        RCLCPP_INFO(logger, "已发布 /scan: points=%zu, scan_time=%.4f", num_readings, duration);
      }
      scan_vec.clear();
      scan_vec_ready = false;
    }
  }

  close(sockfd);
  rclcpp::shutdown();
  return 0;
}
