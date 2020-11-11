// SPDX-License-Identifier: BSD-2-Clause
//
// This code has been produced by the European Spallation Source
// and its partner institutes under the BSD 2 Clause License.
//
// See LICENSE.md at the top level for license information.
//
// Screaming Udder!                              https://esss.se

#include <6s4t_run_stop_generated.h>
#include <chrono>
#include <gtest/gtest.h>
#include <optional>
#include <pl72_run_start_generated.h>

#include "CommandSystem/Parser.h"
#include "Msg.h"
#include "helpers/RunStartStopHelpers.h"

using namespace RunStartStopHelpers;

std::string const InstrumentNameInput = "TEST";
std::string const RunNameInput = "42";
std::string const NexusStructureInput = "{}";
std::string const JobIDInput = "qw3rty";
std::string const CommandIDInput = "some command id";
std::optional<std::string> const ServiceIDInput = "filewriter1";
std::string const BrokerInput = "somehost:1234";
std::string const FilenameInput = "a-dummy-name-01.h5";
uint64_t const StartTimeInput = 123456789000;
uint64_t const StopTimeInput = 123456790000;

class CommandParserHappyStartTests : public testing::Test {
public:
  Command::StartInfo StartInfo;

  void SetUp() override {
    auto MessageBuffer = buildRunStartMessage(
        InstrumentNameInput, RunNameInput, NexusStructureInput, JobIDInput,
        ServiceIDInput, BrokerInput, FilenameInput, StartTimeInput,
        StopTimeInput);

    StartInfo = Command::Parser::extractStartInformation(MessageBuffer);
  }
};

TEST_F(CommandParserHappyStartTests, IfJobIDPresentThenExtractedCorrectly) {
  ASSERT_EQ(JobIDInput, StartInfo.JobID);
}

TEST_F(CommandParserHappyStartTests, IfFilenamePresentThenExtractedCorrectly) {
  ASSERT_EQ(FilenameInput, StartInfo.Filename);
}

TEST_F(CommandParserHappyStartTests, IfBrokerPresentThenExtractedCorrectly) {
  ASSERT_EQ(BrokerInput, StartInfo.BrokerInfo.HostPort);
  ASSERT_EQ(1234u, StartInfo.BrokerInfo.Port);
}

TEST_F(CommandParserHappyStartTests,
       IfNexusStructurePresentThenExtractedCorrectly) {
  ASSERT_EQ(NexusStructureInput, StartInfo.NexusStructure);
}

TEST_F(CommandParserHappyStartTests, IfStartPresentThenExtractedCorrectly) {
  ASSERT_EQ(std::chrono::milliseconds{StartTimeInput}, StartInfo.StartTime);
}

TEST_F(CommandParserHappyStartTests, IfStopPresentThenExtractedCorrectly) {
  ASSERT_EQ(time_point(std::chrono::milliseconds{StopTimeInput}),
            StartInfo.StopTime);
}

TEST_F(CommandParserHappyStartTests, JobIdExtraction) {
  ASSERT_EQ(JobIDInput, StartInfo.JobID);
}

TEST(CommandParserSadStartTests, ThrowsIfNoJobID) {
  std::string const EmptyJobID;
  auto MessageBuffer = buildRunStartMessage(
      InstrumentNameInput, RunNameInput, NexusStructureInput, EmptyJobID,
      ServiceIDInput, BrokerInput, FilenameInput, StartTimeInput,
      StopTimeInput);

  ASSERT_THROW(Command::Parser::extractStartInformation(MessageBuffer),
               std::runtime_error);
}

TEST(CommandParserSadStartTests, ThrowsIfNoFilename) {
  std::string const EmptyFilename;
  auto MessageBuffer = buildRunStartMessage(
      InstrumentNameInput, RunNameInput, NexusStructureInput, JobIDInput,
      ServiceIDInput, BrokerInput, EmptyFilename, StartTimeInput,
      StopTimeInput);

  ASSERT_THROW(Command::Parser::extractStartInformation(MessageBuffer),
               std::runtime_error);
}

TEST(CommandParserSadStartTests, ThrowsIfNoNexusStructure) {
  std::string const EmptyNexusStructure;
  auto MessageBuffer = buildRunStartMessage(
      InstrumentNameInput, RunNameInput, EmptyNexusStructure, JobIDInput,
      ServiceIDInput, BrokerInput, FilenameInput, StartTimeInput,
      StopTimeInput);

  ASSERT_THROW(Command::Parser::extractStartInformation(MessageBuffer),
               std::runtime_error);
}

TEST(CommandParserSadStartTests, IfNoBrokerThenThrows) {
  std::string const EmptyBroker;
  auto MessageBuffer = buildRunStartMessage(
      InstrumentNameInput, RunNameInput, NexusStructureInput, JobIDInput,
      ServiceIDInput, EmptyBroker, FilenameInput, StartTimeInput,
      StopTimeInput);

  ASSERT_THROW(Command::Parser::extractStartInformation(MessageBuffer),
               std::runtime_error);
}

TEST(CommandParserSadStartTests, IfBrokerIsWrongFormThenThrows) {
  std::string const BrokerInvalidFormat = "1234:somehost";
  auto MessageBuffer = buildRunStartMessage(
      InstrumentNameInput, RunNameInput, NexusStructureInput, JobIDInput,
      ServiceIDInput, BrokerInvalidFormat, FilenameInput, StartTimeInput,
      StopTimeInput);

  ASSERT_THROW(Command::Parser::extractStartInformation(MessageBuffer),
               std::runtime_error);
}

TEST(CommandParserStartTests, IfNoStartTimeThenUsesSuppliedCurrentTime) {
  // Start time from flatbuffer is 0 if not supplied when message constructed
  uint64_t const NoStartTime = 0;
  auto MessageBuffer = buildRunStartMessage(
      InstrumentNameInput, RunNameInput, NexusStructureInput, JobIDInput,
      ServiceIDInput, BrokerInput, FilenameInput, NoStartTime, StopTimeInput);

  auto FakeCurrentTime = std::chrono::milliseconds{987654321};

  auto StartInfo =
      Command::Parser::extractStartInformation(MessageBuffer, FakeCurrentTime);

  ASSERT_EQ(FakeCurrentTime, StartInfo.StartTime);
}

TEST(CommandParserStartTests, IfBlankServiceIdThenIsBlank) {
  std::optional<std::string> const EmptyServiceID = "";
  auto MessageBuffer = buildRunStartMessage(
      InstrumentNameInput, RunNameInput, NexusStructureInput, JobIDInput,
      EmptyServiceID, BrokerInput, FilenameInput, StartTimeInput,
      StopTimeInput);

  auto StartInfo = Command::Parser::extractStartInformation(MessageBuffer);

  ASSERT_EQ("", StartInfo.ServiceID);
}

TEST(CommandParserStartTests, IfMissingServiceIdThenIsBlank) {
  std::optional<std::string> const NoServiceID = std::nullopt;
  auto MessageBuffer = buildRunStartMessage(
      InstrumentNameInput, RunNameInput, NexusStructureInput, JobIDInput,
      NoServiceID, BrokerInput, FilenameInput, StartTimeInput, StopTimeInput);

  auto StartInfo = Command::Parser::extractStartInformation(MessageBuffer);

  ASSERT_EQ("", StartInfo.ServiceID);
}

TEST(CommandParserSadStopTests, IfNoJobIdThenThrows) {
  std::string const EmptyJobID;
  auto MessageBuffer = buildRunStopMessage(
      StopTimeInput, RunNameInput, EmptyJobID, CommandIDInput, ServiceIDInput);

  ASSERT_THROW(Command::Parser::extractStopInformation(MessageBuffer),
               std::runtime_error);
}

TEST(CommandParserHappyStopTests, IfJobIdPresentThenExtractedCorrectly) {
  auto MessageBuffer = buildRunStopMessage(
      StopTimeInput, RunNameInput, JobIDInput, CommandIDInput, ServiceIDInput);

  auto StopInfo = Command::Parser::extractStopInformation(MessageBuffer);

  ASSERT_EQ(JobIDInput, StopInfo.JobID);
}

TEST(CommandParserHappyStopTests, IfStopTimePresentThenExtractedCorrectly) {
  auto MessageBuffer = buildRunStopMessage(
      StopTimeInput, RunNameInput, JobIDInput, CommandIDInput, ServiceIDInput);

  auto StopInfo = Command::Parser::extractStopInformation(MessageBuffer);

  ASSERT_EQ(std::chrono::milliseconds{StopTimeInput}, StopInfo.StopTime);
}

TEST(CommandParserStopTests, IfNoServiceIdThenIsBlank) {
  std::optional<std::string> const EmptyServiceID = "";
  auto MessageBuffer = buildRunStopMessage(
      StopTimeInput, RunNameInput, JobIDInput, CommandIDInput, EmptyServiceID);

  auto StopInfo = Command::Parser::extractStopInformation(MessageBuffer);

  ASSERT_EQ("", StopInfo.ServiceID);
}

TEST(CommandParserStopTests, IfMissingServiceIdThenIsBlank) {
  std::optional<std::string> const NoServiceID = std::nullopt;
  auto MessageBuffer = buildRunStopMessage(
      StopTimeInput, RunNameInput, JobIDInput, CommandIDInput, NoServiceID);

  auto StopInfo = Command::Parser::extractStopInformation(MessageBuffer);

  ASSERT_EQ("", StopInfo.ServiceID);
}

TEST(CommandParserStopTests, IfServiceIdPresentThenExtractedCorrectly) {
  auto MessageBuffer = buildRunStopMessage(
      StopTimeInput, RunNameInput, JobIDInput, CommandIDInput, ServiceIDInput);

  auto StopInfo = Command::Parser::extractStopInformation(MessageBuffer);

  ASSERT_EQ(ServiceIDInput, StopInfo.ServiceID);
}

TEST(CommandParserStartTests,
     MessageIsNotStartCommandIfFlatbufferIDNotRecognised) {
  std::string const MessageString = "00001234";
  FileWriter::Msg const TestMessage{MessageString.c_str(),
                                    MessageString.size()};
  ASSERT_FALSE(Command::Parser::isStartCommand(TestMessage));
}

TEST(CommandParserStopTests,
     MessageIsNotStopCommandIfFlatbufferIDNotRecognised) {
  std::string const MessageString = "00001234";
  FileWriter::Msg const TestMessage{MessageString.c_str(),
                                    MessageString.size()};
  ASSERT_FALSE(Command::Parser::isStopCommand(TestMessage));
}

// The following tests for verification of run start/stop messages are disabled
// because it is currently not possible to make verifiable buffers from python.
// This code can be re-enabled if the flatbuffers python library is fixed later.

TEST(CommandParserStartTests,
     DISABLED_MessageIsNotStartCommandIfFlatbufferFailsVerification) {
  std::string const MessageString = fmt::format("0000{}", RunStartIdentifier());
  FileWriter::Msg const TestMessage{MessageString.c_str(),
                                    MessageString.size()};
  ASSERT_FALSE(Command::Parser::isStartCommand(TestMessage));
}

TEST(CommandParserStartTests,
     DISABLED_MessageIsStartCommandIfValidRunStartFlatbuffer) {
  auto MessageBuffer = buildRunStartMessage(
      InstrumentNameInput, RunNameInput, NexusStructureInput, JobIDInput,
      ServiceIDInput, BrokerInput, FilenameInput, StartTimeInput,
      StopTimeInput);
  FileWriter::Msg const TestMessage{MessageBuffer.data(), MessageBuffer.size()};
  ASSERT_TRUE(Command::Parser::isStartCommand(TestMessage));
}

TEST(CommandParserStopTests,
     DISABLED_MessageIsNotStopCommandIfFlatbufferFailsVerification) {
  std::string const MessageString = fmt::format("0000{}", RunStopIdentifier());
  FileWriter::Msg const TestMessage{MessageString.c_str(),
                                    MessageString.size()};
  ASSERT_FALSE(Command::Parser::isStopCommand(TestMessage));
}

TEST(CommandParserStopTests,
     DISABLED_MessageIsStartCommandIfValidRunStopFlatbuffer) {
  auto MessageBuffer = buildRunStopMessage(
      StopTimeInput, RunNameInput, JobIDInput, CommandIDInput, ServiceIDInput);
  FileWriter::Msg const TestMessage{MessageBuffer.data(), MessageBuffer.size()};
  ASSERT_TRUE(Command::Parser::isStopCommand(TestMessage));
}
