// SPDX-License-Identifier: BSD-2-Clause
//
// This code has been produced by the European Spallation Source
// and its partner institutes under the BSD 2 Clause License.
//
// See LICENSE.md at the top level for license information.
//
// Screaming Udder!                              https://esss.se

#pragma once

#include "FlatbufferMessage.h"
#include "WriterModuleBase.h"
#include "WriterModuleConfig/Field.h"
#include <NeXusDataset/NeXusDataset.h>
#include <array>
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <vector>

namespace WriterModule {
namespace s142 {
using FlatbufferMessage = FileWriter::FlatbufferMessage;

using std::string_literals::operator""s;

class SimpleWriter : public WriterModule::Base {
public:
  /// Implements writer module interface.
  InitResult init_hdf(hdf5::node::Group &HDFGroup);
  /// Implements writer module interface.
  void config_post_processing() override;
  /// Implements writer module interface.
  WriterModule::InitResult reopen(hdf5::node::Group &HDFGroup) override;

  /// Write an incoming message which should contain a flatbuffer.
  void write(FlatbufferMessage const &Message) override;

  /// Set the initial value.
  void init_value(std::string const &Value, const uint64_t &Time) override;

  SimpleWriter() : WriterModule::Base(false, "NXlog") {}
  ~SimpleWriter() override = default;

  enum class Type {
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float32,
    float64,
    string
  };

protected:
  SharedLogger Logger = spdlog::get("filewriterlogger");
  std::string findDataType(nlohmann::basic_json<> const &Attribute);

  Type ElementType{Type::float64};

  NeXusDataset::MultiDimDatasetBase NValues;
  NeXusDataset::FixedSizeString SValues;

  /// Timestamps of the f142 updates.
  NeXusDataset::Time Timestamp;

  hdf5::Dimensions Shape{};
  WriterModuleConfig::Field<size_t> ArraySize{this, "array_size", 0};
  WriterModuleConfig::Field<size_t> ChunkSize{this, "chunk_size", 128};
  WriterModuleConfig::Field<std::string> DataType{
      this, std::initializer_list<std::string>({"type"s, "dtype"s}), "double"s};
  WriterModuleConfig::Field<std::string> Unit{
      this, std::initializer_list<std::string>({"value_units"s, "unit"s}), ""s};
};

} // namespace s142
} // namespace WriterModule
