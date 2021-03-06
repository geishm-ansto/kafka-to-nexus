// SPDX-License-Identifier: BSD-2-Clause
//
// This code has been produced by the European Spallation Source
// and its partner institutes under the BSD 2 Clause License.
//
// See LICENSE.md at the top level for license information.
//
// Screaming Udder!                              https://esss.se

#pragma once

namespace Metrics {

struct InternalMetric;

enum struct LogTo { CARBON, LOG_MSG };

class Sink {
public:
  /// Note, access metric values with relaxed memory ordering
  /// (atomic::load(std::memory_order_relaxed)) when implementing reportMetric
  virtual void reportMetric(InternalMetric &MetricToBeReported) = 0;

  /// So that the caller of reportMetric can decide not to give the Sink more to
  /// report on
  virtual bool isHealthy() = 0;

  virtual LogTo getType() = 0;
  virtual ~Sink() = default;
};

} // namespace Metrics
