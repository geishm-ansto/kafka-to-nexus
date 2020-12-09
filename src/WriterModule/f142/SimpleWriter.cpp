// SPDX-License-Identifier: BSD-2-Clause
//
// This code has been produced by the European Spallation Source
// and its partner institutes under the BSD 2 Clause License.
//
// See LICENSE.md at the top level for license information.
//
// Screaming Udder!                              https://esss.se

#include "SimpleWriter.h"
#include "WriterRegistrar.h"
#include "json.h"
#include <algorithm>
#include <cctype>
#include <f142_logdata_generated.h>

namespace WriterModule {
namespace s142 {

using nlohmann::json;

using Type = SimpleWriter::Type;

template <typename Type>
void makeIt(hdf5::node::Group const &Parent, hdf5::Dimensions const &Shape,
            hdf5::Dimensions const &ChunkSize) {
  NeXusDataset::MultiDimDataset<Type>( // NOLINT(bugprone-unused-raii)
      Parent, NeXusDataset::Mode::Create, Shape,
      ChunkSize); // NOLINT(bugprone-unused-raii)
}

template <>
void makeIt<std::string>(hdf5::node::Group const &Parent,
                         hdf5::Dimensions const &, hdf5::Dimensions const &) {
  NeXusDataset::FixedSizeString( // NOLINT(bugprone-unused-raii)
      Parent, "value", NeXusDataset::Mode::Create, 128UL,
      16UL); // NOLINT(bugprone-unused-raii)
}

void initValueDataset(hdf5::node::Group &Parent, Type ElementType,
                      hdf5::Dimensions const &Shape,
                      hdf5::Dimensions const &ChunkSize) {
  using OpenFuncType = std::function<void()>;
  std::map<Type, OpenFuncType> CreateValuesMap{
      {Type::int8, [&]() { makeIt<std::int8_t>(Parent, Shape, ChunkSize); }},
      {Type::uint8, [&]() { makeIt<std::uint8_t>(Parent, Shape, ChunkSize); }},
      {Type::int16, [&]() { makeIt<std::int16_t>(Parent, Shape, ChunkSize); }},
      {Type::uint16,
       [&]() { makeIt<std::uint16_t>(Parent, Shape, ChunkSize); }},
      {Type::int32, [&]() { makeIt<std::int32_t>(Parent, Shape, ChunkSize); }},
      {Type::uint32,
       [&]() { makeIt<std::uint32_t>(Parent, Shape, ChunkSize); }},
      {Type::int64, [&]() { makeIt<std::int64_t>(Parent, Shape, ChunkSize); }},
      {Type::uint64,
       [&]() { makeIt<std::uint64_t>(Parent, Shape, ChunkSize); }},
      {Type::float32,
       [&]() { makeIt<std::float_t>(Parent, Shape, ChunkSize); }},
      {Type::float64,
       [&]() { makeIt<std::double_t>(Parent, Shape, ChunkSize); }},
      {Type::string, [&]() { makeIt<std::string>(Parent, Shape, ChunkSize); }},
  };
  CreateValuesMap.at(ElementType)();
}

/// Parse the configuration for this stream.
void SimpleWriter::config_post_processing() {
  auto ToLower = [](auto InString) {
    std::transform(InString.begin(), InString.end(), InString.begin(),
                   [](auto C) { return std::tolower(C); });
    return InString;
  };
  std::map<std::string, Type> TypeMap{
      {"int8", Type::int8},       {"uint8", Type::uint8},
      {"int16", Type::int16},     {"uint16", Type::uint16},
      {"int32", Type::int32},     {"uint32", Type::uint32},
      {"int64", Type::int64},     {"uint64", Type::uint64},
      {"float32", Type::float32}, {"float64", Type::float64},
      {"float", Type::float32},   {"double", Type::float64},
      {"short", Type::int16},     {"int", Type::int32},
      {"long", Type::int64},      {"string", Type::string}};

  try {
    ElementType = TypeMap.at(ToLower(DataType.getValue()));
  } catch (std::out_of_range &E) {
    Logger->warn("Unknown data type with name \"{}\". Using double.",
                 DataType.getValue());
  }

  if (ArraySize.getValue() > 0) {
    Shape = hdf5::Dimensions{size_t(ArraySize.getValue())};
  }
}

/// \brief Implement the writer module interface, forward to the CREATE case
/// of
/// `init_hdf`.
InitResult SimpleWriter::init_hdf(hdf5::node::Group &HDFGroup) {
  auto Create = NeXusDataset::Mode::Create;
  try {
    NeXusDataset::Time(HDFGroup, Create,
                       ChunkSize); // NOLINT(bugprone-unused-raii)
    initValueDataset(HDFGroup, ElementType, Shape, {ChunkSize});

    if (HDFGroup.attributes.exists("NX_class")) {
      Logger->info("NX_class already specified!");
    } else {
      auto ClassAttribute = HDFGroup.attributes.create<std::string>("NX_class");
      ClassAttribute.write("NXlog");
    }
  } catch (std::exception const &E) {
    auto message = hdf5::error::print_nested(E);
    Logger->error("s142 could not init hdf_parent: {}  trace: {}",
                  static_cast<std::string>(HDFGroup.link().path()), message);
    return InitResult::ERROR;
  }
  if (not Unit.getValue().empty()) {
    HDFGroup["value"].attributes.create_from<std::string>("units", Unit);
  }
  return InitResult::OK;
}

/// \brief Implement the writer module interface, forward to the OPEN case of
/// `init_hdf`.
InitResult SimpleWriter::reopen(hdf5::node::Group &HDFGroup) {
  auto Open = NeXusDataset::Mode::Open;
  try {
    Timestamp = NeXusDataset::Time(HDFGroup, Open);
    if (ElementType == Type::string)
      SValues = NeXusDataset::FixedSizeString(HDFGroup, "value", Open);
    else
      NValues = NeXusDataset::MultiDimDatasetBase(HDFGroup, Open);
  } catch (std::exception &E) {
    Logger->error(
        "Failed to reopen datasets in HDF file with error message: \"{}\"",
        std::string(E.what()));
    return InitResult::ERROR;
  }
  return InitResult::OK;
}

template <typename DataType, class DatasetType>
void appendData(DatasetType &Dataset, const void *Pointer, size_t Size) {
  Dataset.appendArray(
      ArrayAdapter<const DataType>(reinterpret_cast<DataType *>(Pointer), Size),
      {
          Size,
      });
}

template <typename FBValueType, typename ReturnType>
ReturnType extractScalarValue(const LogData *LogDataMessage) {
  auto ScalarValue = LogDataMessage->value_as<FBValueType>();
  return ScalarValue->value();
}

template <typename DataType, typename ValueType, class DatasetType>
void appendScalarData(DatasetType &Dataset, const LogData *LogDataMessage) {
  auto ScalarValue = extractScalarValue<ValueType, DataType>(LogDataMessage);
  Dataset.appendArray(ArrayAdapter<const DataType>(&ScalarValue, 1), {});
}

void SimpleWriter::write(FlatbufferMessage const &Message) {
  auto LogDataMessage = GetLogData(Message.data());
  size_t NrOfElements{1};
  Timestamp.appendElement(LogDataMessage->timestamp());
  auto Type = LogDataMessage->value_type();

  // Note that we are using our knowledge about flatbuffers here to minimise
  // amount of code we have to write by using some pointer arithmetic.
  auto DataPtr = reinterpret_cast<void const *>(
      reinterpret_cast<uint8_t const *>(LogDataMessage->value()) + 4);

  auto extractArrayInfo = [&NrOfElements, &DataPtr]() {
    NrOfElements = *(reinterpret_cast<int const *>(DataPtr) + 1);
    DataPtr = reinterpret_cast<void const *>(
        reinterpret_cast<int const *>(DataPtr) + 2);
  };

  switch (Type) {
  case Value::ArrayByte:
    extractArrayInfo();
    appendData<const std::int8_t>(NValues, DataPtr, NrOfElements);
    break;
  case Value::Byte:
    appendScalarData<const std::int8_t, Byte>(NValues, LogDataMessage);
    break;
  case Value::ArrayUByte:
    extractArrayInfo();
    appendData<const std::uint8_t>(NValues, DataPtr, NrOfElements);
    break;
  case Value::UByte:
    appendScalarData<const std::uint8_t, UByte>(NValues, LogDataMessage);
    break;
  case Value::ArrayShort:
    extractArrayInfo();
    appendData<const std::int16_t>(NValues, DataPtr, NrOfElements);
    break;
  case Value::Short:
    appendScalarData<const std::int16_t, Short>(NValues, LogDataMessage);
    break;
  case Value::ArrayUShort:
    extractArrayInfo();
    appendData<const std::uint16_t>(NValues, DataPtr, NrOfElements);
    break;
  case Value::UShort:
    appendScalarData<const std::uint16_t, UShort>(NValues, LogDataMessage);
    break;
  case Value::ArrayInt:
    extractArrayInfo();
    appendData<const std::int32_t>(NValues, DataPtr, NrOfElements);
    break;
  case Value::Int:
    appendScalarData<const std::int32_t, Int>(NValues, LogDataMessage);
    break;
  case Value::ArrayUInt:
    extractArrayInfo();
    appendData<const std::uint32_t>(NValues, DataPtr, NrOfElements);
    break;
  case Value::UInt:
    appendScalarData<const std::uint32_t, UInt>(NValues, LogDataMessage);
    break;
  case Value::ArrayLong:
    extractArrayInfo();
    appendData<const std::int64_t>(NValues, DataPtr, NrOfElements);
    break;
  case Value::Long:
    appendScalarData<const std::int64_t, Long>(NValues, LogDataMessage);
    break;
  case Value::ArrayULong:
    extractArrayInfo();
    appendData<const std::uint64_t>(NValues, DataPtr, NrOfElements);
    break;
  case Value::ULong:
    appendScalarData<const std::uint64_t, ULong>(NValues, LogDataMessage);
    break;
  case Value::ArrayFloat:
    extractArrayInfo();
    appendData<const float>(NValues, DataPtr, NrOfElements);
    break;
  case Value::Float:
    appendScalarData<const float, Float>(NValues, LogDataMessage);
    break;
  case Value::ArrayDouble:
    extractArrayInfo();
    appendData<const double>(NValues, DataPtr, NrOfElements);
    break;
  case Value::Double:
    appendScalarData<const double, Double>(NValues, LogDataMessage);
    break;
  case Value::String:
    SValues.appendStringElement(
        LogDataMessage->value_as_String()->value()->c_str());
    break;
  default:
    throw WriterModule::WriterException(
        "Unknown data type in f142 flatbuffer.");
  }
}

template <typename DataType, class DatasetType>
void populate(DatasetType &Dataset, std::string const &JsonString) {

  // tranform the json to a buffer and append the values in the buffer
  std::vector<DataType> Buffer;
  auto JsonValue = json::parse(JsonString);
  for (auto &Element : JsonValue) {
    Buffer.emplace_back(Element.get<DataType>());
  }
  size_t Size = Buffer.size();
  auto Shape = Size > 1 ? hdf5::Dimensions{Size} : hdf5::Dimensions{};
  Dataset.appendArray(ArrayAdapter<const DataType>(Buffer.data(), Size), Shape);
}

template <>
void populate<std::string, NeXusDataset::FixedSizeString>(
    NeXusDataset::FixedSizeString &Dataset, std::string const &JsonString) {

  // tranform the json to a buffer and append the values in the buffer
  auto JsonValue = json::parse(JsonString);
  auto Value = JsonValue.get<std::string>();
  Dataset.appendStringElement(Value.c_str());
}

void SimpleWriter::init_value(std::string const &Json, const uint64_t &Time) {

  using OpenFuncType = std::function<void()>;
  std::map<Type, OpenFuncType> PopulateMap{
      {Type::int8, [&]() { populate<std::int8_t>(NValues, Json); }},
      {Type::uint8, [&]() { populate<std::uint8_t>(NValues, Json); }},
      {Type::int16, [&]() { populate<std::int16_t>(NValues, Json); }},
      {Type::uint16, [&]() { populate<std::uint16_t>(NValues, Json); }},
      {Type::int32, [&]() { populate<std::int32_t>(NValues, Json); }},
      {Type::uint32, [&]() { populate<std::uint32_t>(NValues, Json); }},
      {Type::int64, [&]() { populate<std::int64_t>(NValues, Json); }},
      {Type::uint64, [&]() { populate<std::uint64_t>(NValues, Json); }},
      {Type::float32, [&]() { populate<std::float_t>(NValues, Json); }},
      {Type::float64, [&]() { populate<std::double_t>(NValues, Json); }},
      {Type::string, [&]() { populate<std::string>(SValues, Json); }},
  };
  PopulateMap.at(ElementType)();
  Timestamp.appendElement(Time);
}

/// Register the writer module.
static WriterModule::Registry::Registrar<SimpleWriter> RegisterWriter("f142",
                                                                      "s142");

} // namespace s142
} // namespace WriterModule
