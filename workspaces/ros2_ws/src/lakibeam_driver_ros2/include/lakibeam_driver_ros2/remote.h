#pragma once

#include <rclcpp/rclcpp.hpp>

#include <string>

bool sensor_config(
    const std::string &sensor_ipaddr,
    const std::string &parameter,
    const std::string &value,
    const rclcpp::Logger &logger);

bool get_telemetry_data(const std::string &sensor_ipaddr, const rclcpp::Logger &logger);
