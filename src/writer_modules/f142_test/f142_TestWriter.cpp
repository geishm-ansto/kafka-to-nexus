// SPDX-License-Identifier: BSD-2-Clause
//
// This code has been produced by the European Spallation Source
// and its partner institutes under the BSD 2 Clause License.
//
// See LICENSE.md at the top level for license information.
//
// Screaming Udder!                              https://esss.se

#include "f142_TestWriter.h"
#include "HDFFile.h"
#include "WriterRegistrar.h"

namespace WriterModule {
namespace f142 {

/// Parse the configuration for this stream.
void f142_TestWriter::parse_config(std::string const &) {
}

/// \brief Implement the writer module interface, forward to the CREATE case
/// of
/// `init_hdf`.
InitResult f142_TestWriter::init_hdf(hdf5::node::Group &HDFGroup,
                                 std::string const &) {
  auto Create = NeXusDataset::Mode::Create;
  try {
    NeXusDataset::CueIndex(HDFGroup, Create,
                           1024); // NOLINT(bugprone-unused-raii)
  } catch (std::exception const &E) {
    auto message = hdf5::error::print_nested(E);
    return InitResult::ERROR;
  }

  return InitResult::OK;
}

/// \brief Implement the writer module interface, forward to the OPEN case of
/// `init_hdf`.
InitResult f142_TestWriter::reopen(hdf5::node::Group &HDFGroup) {
  auto Open = NeXusDataset::Mode::Open;
  try {
    Counter = NeXusDataset::CueIndex(HDFGroup, Open);
  } catch (std::exception &E) {
    return InitResult::ERROR;
  }
  return InitResult::OK;
}

void f142_TestWriter::write(FlatbufferMessage const &) {
  Counter.appendElement(CounterValue++);
}
/// Register the writer module.
static WriterModule::Registry::Registrar<f142_TestWriter> RegisterWriter("f142",
                                                                     "f142_test");

} // namespace f142
} // namespace WriterModule
