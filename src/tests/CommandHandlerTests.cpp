#include "CommandHandler.h"
#include "MainOpt.h"
#include "helper.h"
#include "json.h"
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>

using nlohmann::json;
using namespace FileWriter;

TEST(CommandHandler_Test, TestFoundBrokerMatchesExpected) {
  auto CommandVector =
      gulp(std::string(TEST_DATA_PATH) + "/msg-cmd-new-01.json");
  ASSERT_EQ(FileWriter::findBroker(
                std::string(CommandVector.data(), CommandVector.size())),
            "localhost:9092");
}

TEST(CommandHandler_Test, TestMissingBrokerSetToDefault) {
  auto CommandVector =
      gulp(std::string(TEST_DATA_PATH) + "/msg-cmd-new-00.json");
  ASSERT_EQ(FileWriter::findBroker(
                std::string(CommandVector.data(), CommandVector.size())),
            "192.168.10.11:9092");
}

class CommandHandler_Testing : public testing::Test {
protected:
  static size_t FileWriterTasksSize(CommandHandler const &CommandHandler) {
    return CommandHandler.getNumberOfFileWriterTasks();
  }
};

TEST_F(CommandHandler_Testing, CatchExceptionOnAttemptToOverwriteFile) {
  std::ofstream ofs;
  ofs.open("tmp-dummy-hdf");
  ofs.close();
  MainOpt MainOpt;
  CommandHandler CommandHandler(MainOpt, nullptr);
  std::string CommandString(R"""(
{
  "cmd": "FileWriter_new",
  "file_attributes": {
    "file_name": "tmp-dummy-hdf"
  },
  "job_id": "vwa98nv983qn98snev",
  "broker": "//localhost:202020",
  "nexus_structure": {
    "children": [
      {
        "type": "group",
        "name": "some_group",
        "children": [
          {
            "type": "dataset",
            "name": "value",
            "values": 42.24
          }
        ]
      }
    ]
  }
})""");
  CommandHandler.handle(
      FileWriter::Msg::owned(CommandString.data(), CommandString.size()));
  // Assert that this command was not accepted by the command handler:
  ASSERT_EQ(CommandHandler_Testing::FileWriterTasksSize(CommandHandler),
            static_cast<size_t>(0));
  unlink("tmp-dummy-hdf");
}

void createFileWithOptionalSWMR(bool UseSWMR) {
  unlink("tmp_swmr_enable.h5");
  MainOpt MainOpt;
  CommandHandler CommandHandler(MainOpt, nullptr);
  std::string CommandString = R"""(
{
  "cmd": "FileWriter_new",
  "file_attributes": {
    "file_name": "tmp_swmr_enable.h5"
  },
  "use_hdf_swmr": true,
  "job_id": "tmp_swmr_enable",
  "broker": "//localhost:202020",
  "nexus_structure": { "children": [] }
})""";
  auto Command = nlohmann::json::parse(CommandString);
  Command["use_hdf_swmr"] = UseSWMR;
  CommandString = Command.dump();
  CommandHandler.handle(
      FileWriter::Msg::owned(CommandString.data(), CommandString.size()));
  ASSERT_EQ(CommandHandler.getNumberOfFileWriterTasks(),
            static_cast<size_t>(1));
  auto &Task = CommandHandler.getFileWriterTaskByJobID("tmp_swmr_enable");
  ASSERT_EQ(Task->hdf_file.isSWMREnabled(), UseSWMR);
  unlink("tmp_swmr_enable.h5");
}

TEST_F(CommandHandler_Testing, OpenFileInNonSWMRMode) {
  createFileWithOptionalSWMR(false);
}

TEST_F(CommandHandler_Testing, OpenFileInSWMRMode) {
  createFileWithOptionalSWMR(true);
}