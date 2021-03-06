// SPDX-License-Identifier: BSD-2-Clause
//
// This code has been produced by the European Spallation Source
// and its partner institutes under the BSD 2 Clause License.
//
// See LICENSE.md at the top level for license information.
//
// Screaming Udder!                              https://esss.se

#pragma once

#include "Kafka/Consumer.h"
#include "MainOpt.h"
#include "Msg.h"
#include "logger.h"

namespace Command {

using FileWriter::Msg;

/// \brief Check for new commands on the topic.
class CommandListener {
public:
  CommandListener(uri::URI CommandTopicUri, Kafka::BrokerSettings Settings);

  /// \brief Poll the Kafka topic for a new command.
  /// \note Will timeout on its first call. Will continue to timeout as long as
  /// it fails to retrieve metadata from the Kafka broker.
  std::pair<Kafka::PollStatus, Msg> pollForCommand();

protected:
  std::string const KafkaAddress;
  std::string const CommandTopic;
  Kafka::BrokerSettings KafkaSettings;
  void setUpConsumer();
  std::unique_ptr<Kafka::ConsumerInterface> Consumer;
  duration CurrentTimeOut;
};
} // namespace Command
