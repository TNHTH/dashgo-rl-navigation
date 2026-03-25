#include "lakibeam_driver_ros2/remote.h"

#include <curl/curl.h>

#include <string>

namespace {
size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
  auto *buffer = static_cast<std::string *>(userp);
  buffer->append(static_cast<char *>(contents), size * nmemb);
  return size * nmemb;
}

bool perform_request(
    const std::string &url,
    const std::string &method,
    const std::string &payload,
    std::string *response,
    const rclcpp::Logger &logger) {
  CURL *curl = curl_easy_init();
  if (curl == nullptr) {
    RCLCPP_ERROR(logger, "curl 初始化失败");
    return false;
  }

  long http_code = 0;
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
  if (!payload.empty()) {
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  }

  const auto result = curl_easy_perform(curl);
  if (result != CURLE_OK) {
    RCLCPP_ERROR(logger, "HTTP 请求失败: %s", curl_easy_strerror(result));
    curl_easy_cleanup(curl);
    return false;
  }

  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_easy_cleanup(curl);
  if (http_code != 200) {
    RCLCPP_WARN(logger, "HTTP 返回码异常: %ld, url=%s", http_code, url.c_str());
    return false;
  }
  return true;
}
}  // namespace

bool sensor_config(
    const std::string &sensor_ipaddr,
    const std::string &parameter,
    const std::string &value,
    const rclcpp::Logger &logger) {
  const std::string url = "http://" + sensor_ipaddr + parameter;
  std::string response;
  const bool ok = perform_request(url, "PUT", value, &response, logger);
  if (ok) {
    RCLCPP_INFO(logger, "已下发雷达参数: %s = %s", url.c_str(), value.c_str());
  }
  return ok;
}

bool get_telemetry_data(const std::string &sensor_ipaddr, const rclcpp::Logger &logger) {
  const std::string firmware_url = "http://" + sensor_ipaddr + "/api/v1/system/firmware";
  const std::string monitor_url = "http://" + sensor_ipaddr + "/api/v1/system/monitor";
  const std::string overview_url = "http://" + sensor_ipaddr + "/api/v1/sensor/overview";

  std::string firmware_response;
  std::string monitor_response;
  std::string overview_response;

  const bool firmware_ok = perform_request(firmware_url, "GET", "", &firmware_response, logger);
  const bool monitor_ok = perform_request(monitor_url, "GET", "", &monitor_response, logger);
  const bool overview_ok = perform_request(overview_url, "GET", "", &overview_response, logger);

  if (firmware_ok) {
    RCLCPP_INFO(logger, "雷达 firmware: %s", firmware_response.c_str());
  }
  if (monitor_ok) {
    RCLCPP_INFO(logger, "雷达 monitor: %s", monitor_response.c_str());
  }
  if (overview_ok) {
    RCLCPP_INFO(logger, "雷达 overview: %s", overview_response.c_str());
  }

  return firmware_ok || monitor_ok || overview_ok;
}
