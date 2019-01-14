#pragma once

#include "KafkaW/KafkaW.h"
#include "MainOpt.h"
#include "URI.h"
#include <thread>

namespace FileWriter {

/// Check for new commands on the topic, return them to the Master.
class CommandListener {
public:
  explicit CommandListener(MainOpt &config);
  ~CommandListener();

  /// Start listening to command messages.
  void start();
  void stop();

  /// Check for new command packets and return one if there is.
  void poll(KafkaW::PollStatus &Status, std::unique_ptr<Msg> Message);

private:
  MainOpt &config;
  std::unique_ptr<KafkaW::ConsumerInterface> consumer;
};
} // namespace FileWriter
