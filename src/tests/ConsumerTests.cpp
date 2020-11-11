// SPDX-License-Identifier: BSD-2-Clause
//
// This code has been produced by the European Spallation Source
// and its partner institutes under the BSD 2 Clause License.
//
// See LICENSE.md at the top level for license information.
//
// Screaming Udder!                              https://esss.se

#include "../Kafka/MetadataException.h"
#include "Kafka/Consumer.h"
#include "helpers/MockMessage.h"
#include "helpers/RdKafkaMocks.h"

#include <gtest/gtest.h>
#include <memory>

using namespace Kafka;
using trompeloeil::_;

class ConsumerTests : public ::testing::Test {
protected:
  void SetUp() override { RdConsumer = std::make_unique<MockKafkaConsumer>(); }

  std::unique_ptr<MockKafkaConsumer> RdConsumer;
};

TEST_F(ConsumerTests, pollReturnsConsumerMessageWithMessagePollStatus) {
  auto *Message = new MockMessage;
  std::string const TestPayload = "Test payload";
  REQUIRE_CALL(*Message, err())
      .TIMES(1)
      .RETURN(RdKafka::ErrorCode::ERR_NO_ERROR);
  // cppcheck-suppress knownArgument
  REQUIRE_CALL(*Message, len()).TIMES(1).RETURN(TestPayload.size());
  RdKafka::MessageTimestamp TimeStamp;
  TimeStamp.timestamp = 1;
  TimeStamp.type = RdKafka::MessageTimestamp::MSG_TIMESTAMP_CREATE_TIME;
  REQUIRE_CALL(*Message, timestamp()).TIMES(2).RETURN(TimeStamp);
  REQUIRE_CALL(*Message, offset()).TIMES(1).RETURN(1);
  ALLOW_CALL(*Message, partition()).RETURN(0);

  REQUIRE_CALL(*Message, payload())
      .TIMES(1)
      .RETURN(
          reinterpret_cast<void *>(const_cast<char *>(TestPayload.c_str())));

  REQUIRE_CALL(*RdConsumer, consume(_)).TIMES(1).RETURN(Message);
  REQUIRE_CALL(*RdConsumer, close()).TIMES(1).RETURN(RdKafka::ERR_NO_ERROR);
  // Put this in scope to call standin destructor
  {
    auto Consumer = std::make_unique<Kafka::Consumer>(
        std::move(RdConsumer),
        std::unique_ptr<RdKafka::Conf>(
            RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL)),
        std::make_unique<Kafka::KafkaEventCb>());
    auto ConsumedMessage = Consumer->poll();
    ASSERT_EQ(ConsumedMessage.first, PollStatus::Message);
  }
}

TEST_F(ConsumerTests, pollReturnsConsumerMessageWithEmptyPollStatusIfTimedOut) {
  auto *Message = new MockMessage;
  REQUIRE_CALL(*Message, err())
      .TIMES(1)
      .RETURN(RdKafka::ErrorCode::ERR__TIMED_OUT);

  REQUIRE_CALL(*RdConsumer, consume(_)).TIMES(1).RETURN(Message);
  REQUIRE_CALL(*RdConsumer, close()).TIMES(1).RETURN(RdKafka::ERR_NO_ERROR);
  // Put this in scope to call standin destructor
  {
    auto Consumer = std::make_unique<Kafka::Consumer>(
        std::move(RdConsumer),
        std::unique_ptr<RdKafka::Conf>(
            RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL)),
        std::make_unique<Kafka::KafkaEventCb>());
    auto ConsumedMessage = Consumer->poll();
    ASSERT_EQ(ConsumedMessage.first, PollStatus::TimedOut);
  }
}

TEST_F(ConsumerTests,
       pollReturnsConsumerMessageWithErrorPollStatusIfUnknownOrUnexpected) {
  auto *Message = new MockMessage;
  REQUIRE_CALL(*Message, err())
      .TIMES(1)
      .RETURN(RdKafka::ErrorCode::ERR__BAD_MSG);

  REQUIRE_CALL(*RdConsumer, consume(_)).TIMES(1).RETURN(Message);
  REQUIRE_CALL(*RdConsumer, close()).TIMES(1).RETURN(RdKafka::ERR_NO_ERROR);
  // Put this in scope to call standin destructor
  {
    auto Consumer = std::make_unique<Kafka::Consumer>(
        std::move(RdConsumer),
        std::unique_ptr<RdKafka::Conf>(
            RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL)),
        std::make_unique<Kafka::KafkaEventCb>());
    auto ConsumedMessage = Consumer->poll();
    ASSERT_EQ(ConsumedMessage.first, PollStatus::Error);
  }
}

#include "Kafka/ConfigureKafka.h"

TEST(ConsumerAssignmentTest, Test1) {
  BrokerSettings SettingsCopy;

  SettingsCopy.KafkaConfiguration.insert({"group.id", "GroupIdStr"});

  SettingsCopy.Address = "broker";

  auto Conf = std::unique_ptr<RdKafka::Conf>(
      RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
  configureKafka(Conf.get(), SettingsCopy);
  auto EventCallback = std::make_unique<KafkaEventCb>();
  std::string ErrorString;
  auto KafkaConsumer = std::unique_ptr<RdKafka::KafkaConsumer>(
      RdKafka::KafkaConsumer::create(Conf.get(), ErrorString));
  auto ConsumerPtr = KafkaConsumer.get();
  if (KafkaConsumer == nullptr) {
    spdlog::get("filewriterlogger")
        ->error("can not create kafka consumer: {}", ErrorString);
    throw std::runtime_error("can not create Kafka consumer");
  }
  auto TestConsumer = std::make_unique<Consumer>(
      std::move(KafkaConsumer), std::move(Conf), std::move(EventCallback));
  std::vector<RdKafka::TopicPartition *> Assignments;
  TestConsumer->addPartitionAtOffset("some_topic1", 0, 0);
  TestConsumer->addPartitionAtOffset("some_topic2", 1, 0);
  ConsumerPtr->assignment(Assignments);

  std::cout << "Size of assignemnts: " << Assignments.size() << std::endl;
  for (auto const &Ass : Assignments) {
    std::cout << "Name of assignemnt: " << Ass->topic() << std::endl;
  }
}
