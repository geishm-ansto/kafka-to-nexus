// SPDX-License-Identifier: BSD-2-Clause
//
// This code has been produced by the European Spallation Source
// and its partner institutes under the BSD 2 Clause License.
//
// See LICENSE.md at the top level for license information.
//
// Screaming Udder!                              https://esss.se

#pragma once

#include <string>
#include <exception>
#include <functional>
#include "Commands.h"
#include "JobListener.h"
#include "CommandListener.h"
#include "ResponseProducer.h"
#include "Msg.h"
#include "JobListener.h"
#include "Kafka/BrokerSettings.h"

namespace Command {

using StartFuncType = std::function<void(StartInfo)>;
using StopTimeFuncType = std::function<void(std::chrono::milliseconds)>;
using StopNowFuncType = std::function<void()>;

class Handler {
public:
  Handler(std::string ServiceIdentifier, Kafka::BrokerSettings const &Settings, uri::URI JobPoolUri, uri::URI CommandTopicUri);
  Handler(std::string ServiceIdentifier, std::unique_ptr<JobListener> JobConsumer, std::unique_ptr<CommandListener> CommandConsumer, std::unique_ptr<ResponseProducer> Response);

  void registerStartFunction(StartFuncType StartFunction);
  void registerSetStopTimeFunction(StopTimeFuncType StopTimeFunction);
  void registerStopNowFunction(StopNowFuncType StopNowFunction);

  void sendHasStoppedMessage();
  void sendErrorEncounteredMessage(std::string ErrorMessage);

  void loopFunction();

private:
  void handleCommand(FileWriter::Msg CommandMsg, bool IgnoreServiceId);
  void handleStartCommand(FileWriter::Msg CommandMsg, bool IgnoreServiceId);
  void handleStopCommand(FileWriter::Msg CommandMsg);
  std::string const ServiceId;
  std::string JobId;
  StartFuncType DoStart{
      [](auto) { throw std::runtime_error("Not implemented."); }};
  StopTimeFuncType DoSetStopTime{
      [](auto) { throw std::runtime_error("Not implemented."); }};
  StopNowFuncType DoStopNow{
      []() { throw std::runtime_error("Not implemented."); }};

  bool PollForJob{true};
  std::unique_ptr<JobListener> JobPool;
  std::unique_ptr<CommandListener> CommandSource;
  std::unique_ptr<ResponseProducer> CommandResponse;
};

} //namespace Command