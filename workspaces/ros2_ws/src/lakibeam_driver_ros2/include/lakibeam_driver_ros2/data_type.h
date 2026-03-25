#pragma once

#include <cstdint>

#define AUTO_ALIGN __attribute__((packed))

#pragma pack(push, 1)
struct MeasuringResult {
  std::uint16_t dist_1;
  std::uint8_t rssi_1;
  std::uint16_t dist_2;
  std::uint8_t rssi_2;
} AUTO_ALIGN;

struct DataBlock {
  std::uint16_t data_flag;
  std::uint16_t azimuth;
  MeasuringResult result[16];
} AUTO_ALIGN;

struct MsopData {
  DataBlock blocks[12];
  std::uint32_t timestamp;
  std::uint16_t factory;
} AUTO_ALIGN;
#pragma pack(pop)

struct ScanResponse {
  std::uint16_t angle;
  std::uint16_t dist;
  std::uint8_t rssi;
  std::uint32_t timestamp;
};
