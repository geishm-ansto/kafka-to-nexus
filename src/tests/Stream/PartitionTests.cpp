// SPDX-License-Identifier: BSD-2-Clause
//
// This code has been produced by the European Spallation Source
// and its partner institutes under the BSD 2 Clause License.
//
// See LICENSE.md at the top level for license information.
//
// Screaming Udder!                              https://esss.se

#include "FlatbufferReader.h"
#include "Stream/Partition.h"
#include "helpers/KafkaMocks.h"
#include "helpers/RdKafkaMocks.h"
#include "helpers/SetExtractorModule.h"
#include <gtest/gtest.h>

using std::chrono_literals::operator""s;
using trompeloeil::_;

class ImmediateExecutor : public IExecutor {
  void sendWork(JobType Task) override { Task(); }
  void sendLowPriorityWork(JobType Task) override { Task(); }
};

class SourceFilterStandInAlt : public Stream::SourceFilter {
public:
  SourceFilterStandInAlt()
      : SourceFilter(std::chrono::system_clock::now(),
                     std::chrono::system_clock::now(), nullptr,
                     Metrics::Registrar("some_reg", {})) {}
  MAKE_MOCK1(filterMessage, bool(FileWriter::FlatbufferMessage &&Message),
             override);
  MAKE_CONST_MOCK0(hasFinished, bool(), override);
};

class zzzzFbReader : public FileWriter::FlatbufferReader {
public:
  bool verify(FileWriter::FlatbufferMessage const &) const override {
    return true;
  }
  std::string
  source_name(FileWriter::FlatbufferMessage const &) const override {
    return zzzzFbReader::UsedSourceName;
  }
  uint64_t timestamp(FileWriter::FlatbufferMessage const &) const override {
    return 1;
  }
  static std::string UsedSourceName;
};
std::string zzzzFbReader::UsedSourceName{"some_name"};

class PartitionStandIn : public Stream::Partition {
public:
  PartitionStandIn(std::unique_ptr<Kafka::ConsumerInterface> Consumer,
                   int Partition, std::string TopicName,
                   Stream::SrcToDst const &Map, Stream::MessageWriter *Writer,
                   Metrics::Registrar RegisterMetric, Stream::time_point Start,
                   Stream::time_point Stop, Stream::duration StopLeeway,
                   Stream::duration KafkaErrorTimeout,
                   std::unique_ptr<IExecutor> Executor)
      : Stream::Partition(std::move(Consumer), Partition, std::move(TopicName),
                          Map, Writer, RegisterMetric, Start, Stop, StopLeeway,
                          KafkaErrorTimeout, std::move(Executor)) {}
  void addPollTask() override {
    // Do nothing as don't want to automatically poll again
  }
  using Partition::ConsumerPtr;
  using Partition::Executor;
  using Partition::FlatbufferErrors;
  using Partition::KafkaErrors;
  using Partition::KafkaTimeouts;
  using Partition::MessagesProcessed;
  using Partition::MessagesReceived;
  using Partition::MsgFilters;
  using Partition::pollForMessage;
  using Partition::processMessage;
  using Partition::StopTime;
  using Partition::StopTimeLeeway;
};

class PartitionTest : public ::testing::Test {
public:
  auto createTestedInstance(Stream::time_point StopTime =
                                std::chrono::system_clock::time_point::max()) {
    Kafka::BrokerSettings BrokerSettingsForTest;
    auto Temp = std::make_unique<PartitionStandIn>(
        std::make_unique<Kafka::MockConsumer>(BrokerSettingsForTest),
        UsedPartitionId, TopicName, UsedMap, nullptr, Registrar, Start,
        StopTime, StopLeeway, ErrorTimeout,
        std::make_unique<ImmediateExecutor>());
    Stop = StopTime;
    Consumer = dynamic_cast<Kafka::MockConsumer *>(Temp->ConsumerPtr.get());
    return Temp;
  }
  Kafka::MockConsumer *Consumer{nullptr};

  int UsedPartitionId{0};
  std::string TopicName{"some_topic"};
  size_t UsedFilterHash{
      FileWriter::calcSourceHash("zzzz", zzzzFbReader::UsedSourceName)};
  Stream::SrcToDst UsedMap{
      Stream::SrcDstKey{UsedFilterHash, nullptr, "some_name", "idid"}};
  Stream::time_point Start{std::chrono::system_clock::now()};
  Stream::time_point Stop{std::chrono::system_clock::time_point::max()};
  Stream::duration StopLeeway{5s};
  Stream::duration ErrorTimeout{10s};
  Metrics::Registrar Registrar{"some_name", {}};
  std::array<char, 9> SomeData{'z', 'z', 'z', 'z', 'z', 'z', 'z', 'z', 'z'};
};

TEST_F(PartitionTest, OnConstructionValuesAreAsExpected) {
  auto StopTime = Start + 20s;
  auto UnderTest = createTestedInstance(StopTime);
  EXPECT_EQ(UnderTest->getPartitionID(), UsedPartitionId);
  EXPECT_EQ(UnderTest->getTopicName(), TopicName);
  EXPECT_EQ(UnderTest->StopTimeLeeway, StopLeeway);
  EXPECT_EQ(UnderTest->StopTime, StopTime);
}

TEST_F(PartitionTest, IfStopTimeTooCloseToMaxThenItIsBackedOff) {
  auto StopTime = std::chrono::system_clock::time_point::max() - StopLeeway / 2;
  auto UnderTest = createTestedInstance(StopTime);
  EXPECT_EQ(UnderTest->StopTime, StopTime - StopLeeway);
}

TEST_F(PartitionTest, EmptyMessageIsIgnored) {
  auto UnderTest = createTestedInstance();
  Kafka::MockConsumer::PollReturnType PollReturn;
  PollReturn.first = Kafka::PollStatus::Empty;
  REQUIRE_CALL(*Consumer, poll()).TIMES(1).LR_RETURN(std::move(PollReturn));
  UnderTest->pollForMessage();
  EXPECT_EQ(int(UnderTest->MessagesReceived), 0);
}

TEST_F(PartitionTest, ActualMessageIsCounted) {
  Kafka::MockConsumer::PollReturnType PollReturn;
  PollReturn.first = Kafka::PollStatus::Message;
  auto UnderTest = createTestedInstance();
  REQUIRE_CALL(*Consumer, poll()).TIMES(1).LR_RETURN(std::move(PollReturn));
  UnderTest->pollForMessage();
  EXPECT_EQ(UnderTest->MessagesReceived.getCounterPtr()->load(), 1);
}

TEST_F(PartitionTest, TimeoutMessageIsCountedButThenIgnored) {
  Kafka::MockConsumer::PollReturnType PollReturn;
  PollReturn.first = Kafka::PollStatus::TimedOut;
  auto UnderTest = createTestedInstance();
  REQUIRE_CALL(*Consumer, poll()).TIMES(1).LR_RETURN(std::move(PollReturn));
  UnderTest->pollForMessage();
  EXPECT_EQ(int(UnderTest->MessagesReceived), 0);
  EXPECT_EQ(int(UnderTest->KafkaTimeouts), 1);
}

TEST_F(PartitionTest, ErrorMessageIsCountedButThenIgnored) {
  Kafka::MockConsumer::PollReturnType PollReturn;
  PollReturn.first = Kafka::PollStatus::Error;
  auto UnderTest = createTestedInstance();
  REQUIRE_CALL(*Consumer, poll()).TIMES(1).LR_RETURN(std::move(PollReturn));
  UnderTest->pollForMessage();
  EXPECT_EQ(UnderTest->MessagesReceived.getCounterPtr()->load(), 0);
  EXPECT_EQ(UnderTest->KafkaErrors.getCounterPtr()->load(), 1);
}

TEST_F(PartitionTest, EndOfPartitionMessageIsIgnored) {
  Kafka::MockConsumer::PollReturnType PollReturn;
  PollReturn.first = Kafka::PollStatus::EndOfPartition;
  auto UnderTest = createTestedInstance();
  REQUIRE_CALL(*Consumer, poll()).TIMES(1).LR_RETURN(std::move(PollReturn));
  UnderTest->pollForMessage();
  EXPECT_EQ(int(UnderTest->MessagesReceived), 0);
}

TEST_F(PartitionTest, WithNoFiltersPartitionIsFinishedOnMessage) {
  Kafka::MockConsumer::PollReturnType PollReturn;
  PollReturn.first = Kafka::PollStatus::Message;
  auto UnderTest = createTestedInstance();
  UnderTest->MsgFilters.clear();
  REQUIRE_CALL(*Consumer, poll()).TIMES(1).LR_RETURN(std::move(PollReturn));
  UnderTest->pollForMessage();
  EXPECT_TRUE(UnderTest->hasFinished());
}

TEST_F(PartitionTest, MessageWithInvalidFlatBufferIsNotProcessed) {
  FileWriter::MessageMetaData MetaData{
      std::chrono::duration_cast<std::chrono::milliseconds>(
          (Start + 10s).time_since_epoch()),
      RdKafka::MessageTimestamp::MSG_TIMESTAMP_CREATE_TIME, 0, 0};
  uint8_t *TempPointer{nullptr};
  Kafka::MockConsumer::PollReturnType PollReturn{
      Kafka::PollStatus::Message, FileWriter::Msg{TempPointer, 0, MetaData}};
  auto UnderTest = createTestedInstance();
  REQUIRE_CALL(*Consumer, poll()).TIMES(1).LR_RETURN(std::move(PollReturn));
  UnderTest->pollForMessage();
  EXPECT_EQ(UnderTest->MessagesReceived.getCounterPtr()->load(), 1);
  EXPECT_EQ(UnderTest->FlatbufferErrors.getCounterPtr()->load(), 1);
}

TEST_F(PartitionTest, MessageWithinStopLeewayDoesNotTriggerFinished) {
  Stop = Start + 20s;
  FileWriter::MessageMetaData MetaData{
      std::chrono::duration_cast<std::chrono::milliseconds>(
          (Stop + StopLeeway).time_since_epoch()),
      RdKafka::MessageTimestamp::MSG_TIMESTAMP_CREATE_TIME, 0, 0};
  uint8_t *TempPointer{nullptr};
  Kafka::MockConsumer::PollReturnType PollReturn{
      Kafka::PollStatus::Message, FileWriter::Msg{TempPointer, 0, MetaData}};
  auto UnderTest = createTestedInstance(Stop);
  REQUIRE_CALL(*Consumer, poll()).TIMES(1).LR_RETURN(std::move(PollReturn));
  UnderTest->pollForMessage();
  EXPECT_FALSE(UnderTest->hasFinished());
}

TEST_F(PartitionTest, MessageAfterStopLeewayTriggersFinished) {
  Stop = Start + 20s;
  FileWriter::MessageMetaData MetaData{
      std::chrono::duration_cast<std::chrono::milliseconds>(
          (Stop + StopLeeway + 1s).time_since_epoch()),
      RdKafka::MessageTimestamp::MSG_TIMESTAMP_CREATE_TIME, 0, 0};
  uint8_t *TempPointer{nullptr};
  Kafka::MockConsumer::PollReturnType PollReturn{
      Kafka::PollStatus::Message, FileWriter::Msg{TempPointer, 0, MetaData}};
  auto UnderTest = createTestedInstance(Stop);
  REQUIRE_CALL(*Consumer, poll()).TIMES(1).LR_RETURN(std::move(PollReturn));
  UnderTest->pollForMessage();
  EXPECT_TRUE(UnderTest->hasFinished());
}

TEST_F(PartitionTest, FiltersAreInitialisedWithOriginalStoptime) {
  auto StopTime = Start + 100s;
  auto UnderTest = createTestedInstance(StopTime);

  for (auto &CFilter : UnderTest->MsgFilters) {
    EXPECT_EQ(CFilter.second->getStopTime(), StopTime);
  }
}

TEST_F(PartitionTest, SetStopTimePropagatesToFilters) {
  auto NewStopTime = Start + 12445s;
  auto UnderTest = createTestedInstance();
  UnderTest->setStopTime(NewStopTime);

  for (auto &CFilter : UnderTest->MsgFilters) {
    EXPECT_EQ(CFilter.second->getStopTime(), NewStopTime);
  }
}

TEST_F(PartitionTest, IfSourceHashUnknownThenNotProcessed) {
  auto UnderTest = createTestedInstance();
  auto TestFilter = std::make_unique<SourceFilterStandInAlt>();
  UnderTest->MsgFilters.clear();
  size_t SomeOtherHash{42};
  UnderTest->MsgFilters[SomeOtherHash] = std::move(TestFilter);
  setExtractorModule<zzzzFbReader>("zzzz");
  FileWriter::Msg Msg(SomeData.data(), SomeData.size());
  UnderTest->processMessage(Msg);
  EXPECT_EQ(UnderTest->MessagesReceived.getCounterPtr()->load(), 1);
  EXPECT_EQ(UnderTest->MessagesProcessed.getCounterPtr()->load(), 0);
}

TEST_F(PartitionTest, IfSourceHashIsKnownThenItIsProcessed) {
  auto UnderTest = createTestedInstance();
  auto TestFilter = std::make_unique<SourceFilterStandInAlt>();
  auto TestFilterPtr = TestFilter.get();
  UnderTest->MsgFilters.at(UsedFilterHash) = std::move(TestFilter);
  REQUIRE_CALL(*TestFilterPtr, filterMessage(_)).TIMES(1).RETURN(true);
  REQUIRE_CALL(*TestFilterPtr, hasFinished()).TIMES(1).RETURN(false);
  setExtractorModule<zzzzFbReader>("zzzz");
  FileWriter::Msg Msg(SomeData.data(), SomeData.size());
  UnderTest->processMessage(Msg);
  EXPECT_EQ(UnderTest->MessagesReceived.getCounterPtr()->load(), 1);
  EXPECT_EQ(UnderTest->MessagesProcessed.getCounterPtr()->load(), 1);
}

TEST_F(PartitionTest, FilterNotRemovedIfNotDone) {
  auto UnderTest = createTestedInstance();
  auto TestFilter = std::make_unique<SourceFilterStandInAlt>();
  auto TestFilterPtr = TestFilter.get();
  auto OldSize = UnderTest->MsgFilters.size();
  UnderTest->MsgFilters.at(UsedFilterHash) = std::move(TestFilter);
  REQUIRE_CALL(*TestFilterPtr, filterMessage(_)).TIMES(1).RETURN(true);
  REQUIRE_CALL(*TestFilterPtr, hasFinished()).TIMES(1).RETURN(false);
  setExtractorModule<zzzzFbReader>("zzzz");
  FileWriter::Msg Msg(SomeData.data(), SomeData.size());
  UnderTest->processMessage(Msg);
  EXPECT_EQ(UnderTest->MsgFilters.size(), OldSize);
}

TEST_F(PartitionTest, FilterIsRemovedWhenDone) {
  auto UnderTest = createTestedInstance();
  auto TestFilter = std::make_unique<SourceFilterStandInAlt>();
  auto TestFilterPtr = TestFilter.get();
  auto OldSize = UnderTest->MsgFilters.size();
  UnderTest->MsgFilters.at(UsedFilterHash) = std::move(TestFilter);
  REQUIRE_CALL(*TestFilterPtr, filterMessage(_)).TIMES(1).RETURN(true);
  REQUIRE_CALL(*TestFilterPtr, hasFinished()).TIMES(1).RETURN(true);
  setExtractorModule<zzzzFbReader>("zzzz");
  FileWriter::Msg Msg(SomeData.data(), SomeData.size());
  UnderTest->processMessage(Msg);
  EXPECT_EQ(UnderTest->MsgFilters.size(), OldSize - 1);
}
