#include "CommandHandler.h"
#include "HDFWriterModule.h"
#include "helper.h"
#include "utils.h"

namespace FileWriter {

// In the future, want to handle many, but not right now.
static int g_N_HANDLED = 0;

CommandHandler::CommandHandler(MainOpt &config, Master *master)
    : config(config), master(master) {
  // Will take care of this in upcoming PR.
  if (false) {
    using namespace rapidjson;
    auto buf1 = gulp("/test/schema-command.json");
    auto doc = make_unique<rapidjson::Document>();
    ParseResult err = doc->Parse(buf1.data(), buf1.size());
    if (err.Code() != ParseErrorCode::kParseErrorNone) {
      LOG(7, "ERROR can not parse schema_command");
      throw std::runtime_error("ERROR can not parse schema_command");
    }
    schema_command.reset(new SchemaDocument(*doc));
  }
}

void CommandHandler::handle_new(rapidjson::Document const &d) {
  // if (g_N_HANDLED > 0) return;
  using namespace rapidjson;
  using std::move;
  using std::string;
  if (schema_command) {
    SchemaValidator vali(*schema_command);
    if (!d.Accept(vali)) {
      StringBuffer sb1, sb2;
      vali.GetInvalidSchemaPointer().StringifyUriFragment(sb1);
      vali.GetInvalidDocumentPointer().StringifyUriFragment(sb2);
      LOG(6, "ERROR command message schema validation:  Invalid schema: {}  "
             "keyword: {}",
          sb1.GetString(), vali.GetInvalidSchemaKeyword());
      return;
    }
  }

  auto fwt = std::unique_ptr<FileWriterTask>(new FileWriterTask);
  std::string fname = "a-dummy-name.h5";
  {
    auto m1 = d.FindMember("file_attributes");
    if (m1 != d.MemberEnd() && m1->value.IsObject()) {
      auto m2 = m1->value.FindMember("file_name");
      if (m2 != m1->value.MemberEnd() && m2->value.IsString()) {
        fname = m2->value.GetString();
      }
    }
  }
  fwt->set_hdf_filename(fname);

  // When FileWriterTask::hdf_init() returns, `stream_hdf_info` will contain
  // the list of streams which have been found in the `nexus_structure`.
  std::vector<StreamHDFInfo> stream_hdf_info;

  std::string jobid = "xxxx-xxxx-xxxx-xxxx";
  {
    auto m = d.FindMember("jobid");
    if (m != d.MemberEnd() && m->value.IsString()) {
      jobid = m->value.GetString();
    }
    else {
      LOG(6, "ERROR command message schema validation:  Invalid schema: {}  "
	  "keyword: {}",
          m->name.GetString());
    }
  }
  // how to handle missing jobid?
  fwt->jobid_init(jobid);

  {
    auto &nexus_structure = d.FindMember("nexus_structure")->value;
    auto x = fwt->hdf_init(nexus_structure, stream_hdf_info);
    if (x) {
      LOG(7, "ERROR hdf init failed, cancel this write command");
      return;
    }
  }

  LOG(6, "Command contains {} streams", stream_hdf_info.size());
  for (auto &stream : stream_hdf_info) {
    auto config_stream_value = get_object(*stream.config_stream, "stream");
    if (!config_stream_value) {
      LOG(5, "Missing stream specification");
      continue;
    }
    auto &config_stream = *config_stream_value.v;
    LOG(7, "Adding stream: {}", json_to_string(config_stream));
    auto topic = get_string(&config_stream, "topic");
    if (!topic) {
      LOG(5, "Missing topic on stream specification");
      continue;
    }
    auto source = get_string(&config_stream, "source");
    if (!source) {
      LOG(5, "Missing source on stream specification");
      continue;
    }
    auto module = get_string(&config_stream, "module");
    if (!module) {
      LOG(5, "Missing module on stream specification");
      continue;
    }

    auto module_factory = HDFWriterModuleRegistry::find(module.v);
    if (!module_factory) {
      LOG(5, "Module '{}' is not available", module.v);
      continue;
    }

    auto hdf_writer_module = module_factory();
    if (!hdf_writer_module) {
      LOG(5, "Can not create a HDFWriterModule for '{}'", module.v);
      continue;
    }

    hdf_writer_module->init_hdf(fwt->hdf_file.h5file, stream.name,
                                config_stream, nullptr);

    auto s = Source(source.v, move(hdf_writer_module));
    fwt->add_source(move(s));
  }

  if (master) {
    ESSTimeStamp start_time(0);
    {
      auto m = d.FindMember("start_time");
      if (m != d.MemberEnd()) {
        start_time = ESSTimeStamp(m->value.GetUint64());
      }
    }
    ESSTimeStamp stop_time(0);
    {
      auto m = d.FindMember("stop_time");
      if (m != d.MemberEnd()) {
        stop_time = ESSTimeStamp(m->value.GetUint64());
      }
    }

    std::string br("localhost:9092");
    auto m = d.FindMember("broker");
    if (m != d.MemberEnd()) {
      auto s = std::string(m->value.GetString());
      if (s.substr(0, 2) == "//") {
        uri::URI u(s);
        br = u.host_port;
      } else {
        // legacy semantics
        br = s;
      }
    }

    auto config_kafka = config.kafka;
    std::vector<std::pair<string, string>> config_kafka_vec;
    for (auto &x : config_kafka) {
      config_kafka_vec.emplace_back(x.first, x.second);
    }

    auto s = std::unique_ptr<StreamMaster<Streamer, DemuxTopic>>(
        new StreamMaster<Streamer, DemuxTopic>(br, std::move(fwt),
                                               config_kafka_vec));
    if (master->status_producer) {
      s->report(master->status_producer, config.status_master_interval);
    }
    if (start_time.count()) {
      LOG(3, "start time :\t{}", start_time.count());
      s->start_time(start_time);
    }
    if (stop_time.count()) {
      LOG(3, "stop time :\t{}", stop_time.count());
      s->stop_time(stop_time);
    }
    s->start();
    master->stream_masters.push_back(std::move(s));
  } else {
    file_writer_tasks.emplace_back(std::move(fwt));
  }
  g_N_HANDLED += 1;
}

void CommandHandler::handle_file_writer_task_clear_all(
    rapidjson::Document const &d) {
  using namespace rapidjson;
  if (master) {
    for (auto &x : master->stream_masters) {
      x->stop();
    }
  }
  file_writer_tasks.clear();
}

void CommandHandler::handle_exit(rapidjson::Document const &d) {
  if (master)
    master->stop();
}

void CommandHandler::handle_stream_master_stop(rapidjson::Document const &d) {
  // parse document to get jobid
  auto s = get_string(&d, "jobid");
  auto jobid = std::string(s);
  if (master) {
    for (auto &x : master->stream_masters) {
      x->stop(jobid);
    }
  }
  // TODO
  // remove task from file_writer_tasks
}

void CommandHandler::handle(rapidjson::Document const &d) {
  using std::string;
  using namespace rapidjson;
  uint64_t teamid = 0;
  uint64_t cmd_teamid = 0;
  if (master) {
    teamid = master->config.teamid;
  }
  if (auto i = get_int(&d, "teamid")) {
    cmd_teamid = int64_t(i);
  }
  if (cmd_teamid != teamid) {
    LOG(1, "INFO command is for teamid {:016x}, we are {:016x}", cmd_teamid,
        teamid);
    return;
  }

  // The ways to give commands will be unified in upcoming PR.
  if (auto s = get_string(&d, "cmd")) {
    auto cmd = string(s);
    if (cmd == "FileWriter_new") {
      handle_new(d);
      return;
    }
    if (cmd == "FileWriter_exit") {
      handle_exit(d);
      return;
    }
    if (cmd == "FileWriter_stop") {
      handle_exit(d);
      return;
    }
  }

  if (auto s = get_string(&d, "recv_type")) {
    auto recv_type = string(s);
    if (recv_type == "FileWriter") {
      if (auto s = get_string(&d, "cmd")) {
        auto cmd = string(s);
        if (cmd == "file_writer_tasks_clear_all") {
          handle_file_writer_task_clear_all(d);
          return;
        }
      }
    }
  }

  StringBuffer buffer;
  PrettyWriter<StringBuffer> writer(buffer);
  d.Accept(writer);
  LOG(3, "ERROR could not figure out this command: {}", buffer.GetString());
}

void CommandHandler::handle(Msg const &msg) {
  using std::string;
  using namespace rapidjson;
  auto doc = make_unique<Document>();
  ParseResult err = doc->Parse((char *)msg.data, msg.size);
  if (doc->HasParseError()) {
    LOG(2, "ERROR json parse: {} {}", err.Code(), GetParseError_En(err.Code()));
    return;
  }
  handle(*doc);
}

} // namespace FileWriter
