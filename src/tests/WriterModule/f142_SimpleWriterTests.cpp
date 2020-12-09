// SPDX-License-Identifier: BSD-2-Clause
//
// This code has been produced by the European Spallation Source
// and its partner institutes under the BSD 2 Clause License.
//
// See LICENSE.md at the top level for license information.
//
// Screaming Udder!                              https://esss.se

// This filename is chosen such that it shows up in searches after the
// case-sensitive flatbuffer schema identifier.

#include <gtest/gtest.h>
#include <h5cpp/hdf5.hpp>
#include <memory>

#include "AccessMessageMetadata/f142/f142_Extractor.h"
#include "FlatbufferMessage.h"
#include "WriterModule/f142/SimpleWriter.h"
#include "helper.h"
#include "helpers/HDFFileTestHelper.h"
#include "helpers/SetExtractorModule.h"
#include <f142_logdata_generated.h>

using nlohmann::json;

using namespace WriterModule::s142;

class s142Init : public ::testing::Test {
public:
  void SetUp() override {
    TestFile =
        HDFFileTestHelper::createInMemoryTestFile("SomeTestFile.hdf5", false);
    RootGroup = TestFile->hdfGroup();
  }
  std::unique_ptr<HDFFileTestHelper::DebugHDFFile> TestFile;
  hdf5::node::Group RootGroup;
};

class SimpleWriterStandIn : public SimpleWriter {
public:
  using SimpleWriter::Shape;
  using SimpleWriter::ChunkSize;
  using SimpleWriter::ElementType;
  using SimpleWriter::Timestamp;
  using SimpleWriter::NValues;
  using SimpleWriter::SValues;
};

TEST_F(s142Init, BasicDefaultInit) {
  SimpleWriter TestWriter;
  TestWriter.init_hdf(RootGroup);
  EXPECT_TRUE(RootGroup.has_dataset("time"));
  EXPECT_TRUE(RootGroup.has_dataset("value"));
}

TEST_F(s142Init, ReOpenSuccess) {
  SimpleWriter TestWriter;
  TestWriter.init_hdf(RootGroup);
  EXPECT_EQ(TestWriter.reopen(RootGroup), WriterModule::InitResult::OK);
}

TEST_F(s142Init, ReOpenFailure) {
  SimpleWriter TestWriter;
  EXPECT_EQ(TestWriter.reopen(RootGroup), WriterModule::InitResult::ERROR);
}

TEST_F(s142Init, CheckInitDataType) {
  SimpleWriterStandIn TestWriter;
  TestWriter.init_hdf(RootGroup);
  auto Open = NeXusDataset::Mode::Open;
  NeXusDataset::MultiDimDatasetBase Value(RootGroup, Open);
  EXPECT_EQ(Value.datatype(), hdf5::datatype::create<double>());
}

TEST_F(s142Init, CheckValueInitShape1) {
  SimpleWriterStandIn TestWriter;
  TestWriter.init_hdf(RootGroup);
  auto Open = NeXusDataset::Mode::Open;
  NeXusDataset::MultiDimDatasetBase Value(RootGroup, Open);
  EXPECT_EQ(hdf5::Dimensions({0}), Value.get_extent());
}

TEST_F(s142Init, CheckValueInitShape2) {
  SimpleWriterStandIn TestWriter;
  TestWriter.Shape = hdf5::Dimensions{10};
  TestWriter.init_hdf(RootGroup);
  auto Open = NeXusDataset::Mode::Open;
  NeXusDataset::MultiDimDatasetBase Value(RootGroup, Open);
  EXPECT_EQ(hdf5::Dimensions({0, 10}), Value.get_extent());
}

TEST_F(s142Init, CheckAllNumericDataTypes) {
  std::vector<std::pair<SimpleWriter::Type, hdf5::datatype::Datatype>> TypeMap{
      {SimpleWriter::Type::int8, hdf5::datatype::create<std::int8_t>()},
      {SimpleWriter::Type::uint8, hdf5::datatype::create<std::uint8_t>()},
      {SimpleWriter::Type::int16, hdf5::datatype::create<std::int16_t>()},
      {SimpleWriter::Type::uint16, hdf5::datatype::create<std::uint16_t>()},
      {SimpleWriter::Type::int32, hdf5::datatype::create<std::int32_t>()},
      {SimpleWriter::Type::uint32, hdf5::datatype::create<std::uint32_t>()},
      {SimpleWriter::Type::int64, hdf5::datatype::create<std::int64_t>()},
      {SimpleWriter::Type::uint64, hdf5::datatype::create<std::uint64_t>()},
      {SimpleWriter::Type::float32, hdf5::datatype::create<float>()},
      {SimpleWriter::Type::float64, hdf5::datatype::create<double>()}};
  auto Open = NeXusDataset::Mode::Open;
  SimpleWriterStandIn TestWriter;
  int Ctr{0};
  for (auto &Type : TypeMap) {
    auto CurrentGroup = RootGroup.create_group("Group" + std::to_string(Ctr++));
    TestWriter.ElementType = Type.first;
    TestWriter.init_hdf(CurrentGroup);
    NeXusDataset::MultiDimDatasetBase Value(CurrentGroup, Open);
    EXPECT_EQ(Type.second, Value.datatype());
  }
}

TEST_F(s142Init, CheckStringDataType) {
  auto Open = NeXusDataset::Mode::Open;
  SimpleWriterStandIn TestWriter;
  TestWriter.ElementType = SimpleWriter::Type::string;
  std::string Ctr{"Stringy"};
  auto CurrentGroup = RootGroup.create_group("Group" + Ctr);
  TestWriter.init_hdf(CurrentGroup);
  NeXusDataset::FixedSizeString Value(CurrentGroup, "value", Open);
  auto TestType = Value.datatype();
  hdf5::datatype::String StringType(
      hdf5::datatype::String::fixed(TestType.size()));
  StringType.encoding(hdf5::datatype::CharacterEncoding::UTF8);
  StringType.padding(hdf5::datatype::StringPad::NULLTERM);
  EXPECT_EQ(StringType, TestType);
}

class s142ConfigParse : public ::testing::Test {
public:
};

TEST_F(s142ConfigParse, EmptyConfig) {
  SimpleWriterStandIn TestWriter;
  TestWriter.parse_config("{}");
  SimpleWriterStandIn TestWriter2;
  EXPECT_EQ(TestWriter.ElementType, TestWriter2.ElementType);
  EXPECT_EQ(TestWriter.Shape, TestWriter2.Shape);
  EXPECT_EQ(TestWriter.ChunkSize, TestWriter2.ChunkSize);
}

TEST_F(s142ConfigParse, SetArraySize) {
  SimpleWriterStandIn TestWriter;
  TestWriter.parse_config(R"({
              "array_size": 3
            })");
  SimpleWriterStandIn TestWriter2;
  EXPECT_EQ(TestWriter.Shape, hdf5::Dimensions{3u});
  EXPECT_EQ(TestWriter.ElementType, TestWriter2.ElementType);
  EXPECT_EQ(TestWriter.ChunkSize, TestWriter2.ChunkSize);
}

TEST_F(s142ConfigParse, SetChunkSize) {
  SimpleWriterStandIn TestWriter;
  TestWriter.parse_config(R"({
              "chunk_size": 511
            })");
  SimpleWriterStandIn TestWriter2;
  EXPECT_EQ(TestWriter.Shape, TestWriter2.Shape);
  EXPECT_EQ(TestWriter.ElementType, TestWriter2.ElementType);
  EXPECT_EQ(TestWriter.ChunkSize.getValue(), 511u);
}

TEST_F(s142ConfigParse, CueInterval) {
  SimpleWriterStandIn TestWriter;
  TestWriter.parse_config(R"({
              "cue_interval": 24
            })");
  SimpleWriterStandIn TestWriter2;
  EXPECT_EQ(TestWriter.Shape, TestWriter2.Shape);
  EXPECT_EQ(TestWriter.ElementType, TestWriter2.ElementType);
  EXPECT_EQ(TestWriter.ChunkSize, TestWriter2.ChunkSize);
}

TEST_F(s142ConfigParse, DataType1) {
  SimpleWriterStandIn TestWriter;
  TestWriter.parse_config(R"({
              "type": "int8"
            })");
  SimpleWriterStandIn TestWriter2;
  EXPECT_EQ(TestWriter.Shape, TestWriter2.Shape);
  EXPECT_EQ(TestWriter.ElementType, SimpleWriter::Type::int8);
  EXPECT_EQ(TestWriter.ChunkSize, TestWriter2.ChunkSize);
}

TEST_F(s142ConfigParse, DataType2) {
  SimpleWriterStandIn TestWriter;
  TestWriter.parse_config(R"({
              "dtype": "uint64"
            })");
  SimpleWriterStandIn TestWriter2;
  EXPECT_EQ(TestWriter.Shape, TestWriter2.Shape);
  EXPECT_EQ(TestWriter.ElementType, SimpleWriter::Type::uint64);
  EXPECT_EQ(TestWriter.ChunkSize, TestWriter2.ChunkSize);
}

TEST_F(s142ConfigParse, DataTypeFailure) {
  SimpleWriterStandIn TestWriter;
  TestWriter.parse_config(R"({
              "Dtype": "uint64"
            })");
  SimpleWriterStandIn TestWriter2;
  EXPECT_EQ(TestWriter.Shape, TestWriter2.Shape);
  EXPECT_EQ(TestWriter.ElementType, SimpleWriter::Type::float64);
  EXPECT_EQ(TestWriter.ChunkSize, TestWriter2.ChunkSize);
}

TEST_F(s142ConfigParse, DataTypes) {
  using Type = SimpleWriter::Type;
  std::vector<std::pair<std::string, Type>> TypeList{
      {"int8", Type::int8},       {"INT8", Type::int8},
      {"SHORT", Type::int16},     {"UINT8", Type::uint8},
      {"INT16", Type::int16},     {"Uint16", Type::uint16},
      {"int32", Type::int32},     {"Int", Type::int32},
      {"uint32", Type::uint32},   {"int64", Type::int64},
      {"long", Type::int64},      {"uint64", Type::uint64},
      {"float32", Type::float32}, {"float", Type::float32},
      {"FLOAT", Type::float32},   {"float64", Type::float64},
      {"double", Type::float64},  {"DOUBLE", Type::float64},
      {"string", Type::string},   {"STRING", Type::string}};
  for (auto &CType : TypeList) {
    SimpleWriterStandIn TestWriter;
    EXPECT_EQ(TestWriter.ElementType, Type::float64);
    TestWriter.parse_config("{\"type\":\"" + CType.first + "\"}");
    EXPECT_EQ(TestWriter.ElementType, CType.second)
        << "Failed on type string: " << CType.first;
  }
}

class s142WriteData : public ::testing::Test {
public:
  void SetUp() override {
    TestFile =
        HDFFileTestHelper::createInMemoryTestFile("SomeTestFile.hdf5", false);
    RootGroup = TestFile->hdfGroup();
    setExtractorModule<AccessMessageMetadata::f142_Extractor>("f142");
  }
  std::unique_ptr<HDFFileTestHelper::DebugHDFFile> TestFile;
  hdf5::node::Group RootGroup;
};

template <class ValFuncType>
std::pair<std::unique_ptr<uint8_t[]>, size_t> generateFBufferMessageBase(
    ValFuncType ValueFunc, Value ValueTypeId, std::uint64_t Timestamp) {
  auto Builder = flatbuffers::FlatBufferBuilder();
  auto SourceNameOffset = Builder.CreateString("SomeSourceName");
  auto ValueOffset = ValueFunc(Builder);
  LogDataBuilder LogDataBuilder(Builder);
  LogDataBuilder.add_value(ValueOffset);
  LogDataBuilder.add_timestamp(Timestamp);
  LogDataBuilder.add_source_name(SourceNameOffset);
  LogDataBuilder.add_value_type(ValueTypeId);

  FinishLogDataBuffer(Builder, LogDataBuilder.Finish());
  size_t BufferSize = Builder.GetSize();
  auto ReturnBuffer = std::make_unique<uint8_t[]>(BufferSize);
  std::memcpy(ReturnBuffer.get(), Builder.GetBufferPointer(), BufferSize);
  return {std::move(ReturnBuffer), BufferSize};
}

std::pair<std::unique_ptr<uint8_t[]>, size_t> generateFBufferMessage(
    double Value, std::uint64_t Timestamp) {
  auto ValueFunc = [Value](auto &Builder) {
    DoubleBuilder ValueBuilder(Builder);
    ValueBuilder.add_value(Value);
    return ValueBuilder.Finish().Union();
  };
  return generateFBufferMessageBase(ValueFunc, Value::Double, Timestamp);
}

std::pair<std::unique_ptr<uint8_t[]>, size_t>
generateFBMessage(std::string Value, std::uint64_t Timestamp) {
  auto ValueFunc = [Value](auto &Builder) {
    auto svalue = Builder.CreateString(Value);
    StringBuilder ValueBuilder(Builder);
    ValueBuilder.add_value(svalue);
    return ValueBuilder.Finish().Union();
  };
  return generateFBufferMessageBase(ValueFunc, Value::String, Timestamp);
}

TEST_F(s142WriteData, ConfigUnitsAttributeOnValueDataset) {
  SimpleWriterStandIn TestWriter;
  const std::string units_string = "parsecs";
  // GIVEN value_units is specified in the JSON config
  TestWriter.parse_config(
      fmt::format(R"({{"value_units": "{}"}})", units_string));

  // WHEN the writer module creates the datasets
  TestWriter.init_hdf(RootGroup);
  TestWriter.reopen(RootGroup);

  // THEN a units attributes is created on the value dataset with the specified
  // string
  std::string attribute_value;
  EXPECT_NO_THROW(TestWriter.NValues.attributes["units"].read(attribute_value))
      << "Expect units attribute to be present on the value dataset";
  EXPECT_EQ(attribute_value, units_string) << "Expect units attribute to have "
                                              "the value specified in the JSON "
                                              "configuration";
}

TEST_F(s142WriteData, ConfigUnitsAttributeOnValueDatasetIfEmpty) {
  SimpleWriterStandIn TestWriter;
  // GIVEN value_units is specified as an empty string in the JSON config
  TestWriter.parse_config(R"({"value_units": ""})");

  // WHEN the writer module creates the datasets
  TestWriter.init_hdf(RootGroup);
  TestWriter.reopen(RootGroup);

  EXPECT_FALSE(TestWriter.NValues.attributes.exists("units"))
      << "units attribute should not be created if the config string is empty";
}

TEST_F(s142WriteData, UnitsAttributeOnValueDatasetNotCreatedIfNotInConfig) {
  SimpleWriterStandIn TestWriter;
  // GIVEN value_units is not specified in the JSON config
  TestWriter.parse_config("{}");

  // WHEN the writer module creates the datasets
  TestWriter.init_hdf(RootGroup);
  TestWriter.reopen(RootGroup);

  // THEN a units attributes is not created on the value dataset
  EXPECT_FALSE(TestWriter.NValues.attributes.exists("units"))
      << "units attribute should not be created if it was not specified in the "
         "JSON config";
}

TEST_F(s142WriteData, InitValueTest) {
  SimpleWriterStandIn TestWriter;
  TestWriter.parse_config(R"({
              "type": "int16"
            })");
  std::uint64_t Timestamp{12};
  std::int16_t ElementValue{12345};
  TestWriter.init_hdf(RootGroup);
  TestWriter.reopen(RootGroup);
  EXPECT_EQ(TestWriter.ElementType, SimpleWriter::Type::int16);
  TestWriter.init_value(R"(12345)", Timestamp);
  ASSERT_EQ(TestWriter.NValues.get_extent(), hdf5::Dimensions({1}));
  ASSERT_EQ(TestWriter.Timestamp.dataspace().size(), 1);
  std::vector<std::int16_t> WrittenValues(1);
  TestWriter.NValues.read(WrittenValues);
  EXPECT_EQ(WrittenValues.at(0), ElementValue);
  std::vector<std::uint64_t> WrittenTimes(1);
  TestWriter.Timestamp.read(WrittenTimes);
  EXPECT_EQ(WrittenTimes.at(0), Timestamp);
}

TEST_F(s142WriteData, WriteOneElement) {
  SimpleWriterStandIn TestWriter;
  TestWriter.init_hdf(RootGroup);
  TestWriter.reopen(RootGroup);
  double ElementValue{3.14};
  std::uint64_t Timestamp{11};
  auto FlatbufferData = generateFBufferMessage(ElementValue, Timestamp);
  EXPECT_EQ(TestWriter.NValues.get_extent(), hdf5::Dimensions({0}));
  EXPECT_EQ(TestWriter.Timestamp.dataspace().size(), 0);
  TestWriter.write(FileWriter::FlatbufferMessage(FlatbufferData.first.get(),
                                                 FlatbufferData.second));
  ASSERT_EQ(TestWriter.NValues.get_extent(), hdf5::Dimensions({1}));
  ASSERT_EQ(TestWriter.Timestamp.dataspace().size(), 1);
  std::vector<double> WrittenValues(1);
  TestWriter.NValues.read(WrittenValues);
  EXPECT_EQ(WrittenValues.at(0), ElementValue);
  std::vector<std::uint64_t> WrittenTimes(1);
  TestWriter.Timestamp.read(WrittenTimes);
  EXPECT_EQ(WrittenTimes.at(0), Timestamp);
}

TEST_F(s142WriteData, WriteOneDefaultValueElement) {
  SimpleWriterStandIn TestWriter;
  TestWriter.init_hdf(RootGroup);
  TestWriter.reopen(RootGroup);
  // 0 is the default value for a number in flatbuffers, so it doesn't actually
  // end up in buffer. We'll test this specifically, because it has
  // caused a bug in the past.
  double ElementValue{0.0};
  std::uint64_t Timestamp{11};
  auto FlatbufferData = generateFBufferMessage(ElementValue, Timestamp);
  EXPECT_EQ(TestWriter.NValues.get_extent(), hdf5::Dimensions({0}));
  EXPECT_EQ(TestWriter.Timestamp.dataspace().size(), 0);
  TestWriter.write(FileWriter::FlatbufferMessage(FlatbufferData.first.get(),
                                                 FlatbufferData.second));
  ASSERT_EQ(TestWriter.NValues.get_extent(), hdf5::Dimensions({1}));
  ASSERT_EQ(TestWriter.Timestamp.dataspace().size(), 1);
  std::vector<double> WrittenValues(1);
  TestWriter.NValues.read(WrittenValues);
  EXPECT_EQ(WrittenValues.at(0), ElementValue);
  std::vector<std::uint64_t> WrittenTimes(1);
  TestWriter.Timestamp.read(WrittenTimes);
  EXPECT_EQ(WrittenTimes.at(0), Timestamp);
}

std::pair<std::unique_ptr<uint8_t[]>, size_t>
generateFBufferArrayMessage(std::vector<double> Value, uint64_t Timestamp) {
  auto ValueFunc = [Value](auto &Builder) {
    auto VectorOffset = Builder.CreateVector(Value);
    ArrayDoubleBuilder ValueBuilder(Builder);
    ValueBuilder.add_value(VectorOffset);
    return ValueBuilder.Finish().Union();
  };
  return generateFBufferMessageBase(ValueFunc, Value::ArrayDouble,
                                       Timestamp);
}

TEST_F(s142WriteData, WriteOneArray) {
  SimpleWriterStandIn TestWriter;
  TestWriter.parse_config(R"({
            "array_size": 3
          })");
  TestWriter.init_hdf(RootGroup);
  TestWriter.reopen(RootGroup);
  std::vector<double> ElementValues{3.14, 4.5, 3.1};
  uint64_t Timestamp{12};
  auto FlatbufferData =
      generateFBufferArrayMessage(ElementValues, Timestamp);
  TestWriter.write(FileWriter::FlatbufferMessage(FlatbufferData.first.get(),
                                                 FlatbufferData.second));
  ASSERT_EQ(TestWriter.NValues.get_extent(), hdf5::Dimensions({1, 3}));
  std::vector<double> WrittenValues(3);
  TestWriter.NValues.read(WrittenValues);
  EXPECT_EQ(WrittenValues, ElementValues);
}


TEST_F(s142WriteData, InitSValueTest) {
  SimpleWriterStandIn TestWriter;
  TestWriter.parse_config(R"({
              "type": "string"
            })");
  std::uint64_t Timestamp{0};
  std::string ElementValue{"1234567890"};
  TestWriter.init_hdf(RootGroup);
  TestWriter.reopen(RootGroup);
  EXPECT_EQ(TestWriter.ElementType, SimpleWriter::Type::string);
  TestWriter.init_value(R"("1234567890")", Timestamp);
  std::string ReadBackString;
  TestWriter.SValues.read(ReadBackString, TestWriter.SValues.datatype(),
                   hdf5::dataspace::Scalar(),
                   hdf5::dataspace::Hyperslab{{0}, {1}});
  std::string CompareString(ReadBackString.data());
  EXPECT_EQ(ElementValue, CompareString);
}

TEST_F(s142WriteData, WriteTwoStringElements) {
  using Type = SimpleWriterStandIn::Type;
  SimpleWriterStandIn TestWriter;
  TestWriter.ElementType = Type::string;
  TestWriter.init_hdf(RootGroup);
  TestWriter.reopen(RootGroup);
  std::string TestString1{"hello world"};
  std::uint64_t Timestamp1{11};
  auto FlatbufferData = generateFBMessage(TestString1, Timestamp1);
  TestWriter.write(FileWriter::FlatbufferMessage(FlatbufferData.first.get(),
                                                 FlatbufferData.second));
  std::string TestString2{"another world"};
  std::uint64_t Timestamp{13};
  FlatbufferData = generateFBMessage(TestString2, Timestamp);
  TestWriter.write(FileWriter::FlatbufferMessage(FlatbufferData.first.get(),
                                                 FlatbufferData.second));
  auto comparison = [&](size_t Index, auto InString) {
    std::string ReadBackString;
    TestWriter.SValues.read(ReadBackString, TestWriter.SValues.datatype(),
                    hdf5::dataspace::Scalar(),
                    hdf5::dataspace::Hyperslab{{Index}, {1}});
    std::string CompareString(ReadBackString.data());
    EXPECT_EQ(CompareString, InString);
  };
  comparison(0, TestString1);
  comparison(1, TestString2);
}
