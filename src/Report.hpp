#pragma once

#include <chrono>
#include <memory>
#include <thread>

#include "Status.hpp"
#include "KafkaW.h"
#include "StatusWriter.hpp"
#include "logger.h"

namespace KafkaW {
class ProducerTopic;
}

namespace FileWriter {

class Report {
  using SMEC = Status::StreamMasterErrorCode;

public:
  Report() {}
  Report(std::shared_ptr<KafkaW::ProducerTopic> producer,
         const int delay = 1000)
      : report_producer_{producer}, delay_{std::chrono::milliseconds(delay)} {}
  Report(const Report &other) = delete;
  Report(Report &&other) = default;
  Report &operator=(Report &&other) = default;
  ~Report() = default;

  template <class S>
  void report(S &streamer, std::atomic<bool> &stop,
              std::atomic<SMEC> &stream_master_status) {
  while (!stop.load()) {
    std::this_thread::sleep_for(delay_);
    auto error = produce_single_report(streamer, stream_master_status.load());
    if (error == SMEC::report_failure) {
      stream_master_status = error;
      return;
    }
  }
  std::this_thread::sleep_for(delay_);
  auto error = produce_single_report(streamer, stream_master_status.load());
  if (error != SMEC::no_error) { // termination message
    stream_master_status = error;
  }
  return;
  }

private:
  template <class S>
  SMEC produce_single_report(S &streamer, SMEC stream_master_status) {
    if (!report_producer_) {
      LOG(1, "ProucerTopic error: can't produce StreamMaster status report");
      return SMEC::report_failure;
    }
    
    Status::StreamMasterInfo info;
    info.time(delay_);
    for (auto &s : streamer) {
      info.add(s.first, s.second.info());
    }
    auto value = Status::pprint<Status::JSONStreamWriter>(info);
    report_producer_->produce(reinterpret_cast<unsigned char *>(&value[0]),
			      value.size());  
    return SMEC::no_error;
  }

  std::shared_ptr<KafkaW::ProducerTopic> report_producer_{nullptr};
  std::chrono::milliseconds delay_{milliseconds(1000)};
};
} // namespace FileWriter
