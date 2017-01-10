#include <cstdlib>
#include <cstdio>
#include <string>
#include <getopt.h>
#include "logger.h"
#include "Master.h"

#if HAVE_GTEST
#include <gtest/gtest.h>
#endif

extern "C" char const GIT_COMMIT[];



// POD
struct MainOpt {
bool help = false;
bool verbose = false;
BrightnESS::FileWriter::MasterConfig master_config;
};


int main(int argc, char ** argv) {
	#if HAVE_GTEST
	// In the current stage, it makes sense for tests to live here.
	// Before any possible changes to MainOpt.
	if (argc == 2 and strcmp("--test", argv[1]) == 0) {
		log_level = 1;
		::testing::InitGoogleTest(&argc, argv);
		return RUN_ALL_TESTS();
	}
	#endif

	MainOpt opt;

	static struct option long_options[] = {
		{"help",                            no_argument,              0, 'h'},
		{"broker-command-address",          required_argument,        0,  0 },
		{"broker-command-topic",            required_argument,        0,  0 },
		{"verbose",                         no_argument,              0, 'v'},
		{0, 0, 0, 0},
	};
	std::string cmd;
	int option_index = 0;
	bool getopt_error = false;
	while (true) {
		int c = getopt_long(argc, argv, "vh", long_options, &option_index);
		//LOG(5, "c getopt {}", c);
		if (c == -1) break;
		if (c == '?') {
			getopt_error = true;
		}
		switch (c) {
		case 'v':
			// Do nothing, purpose is to fall through to long-option handling
			LOG(9, "Verbose");
			opt.verbose = true;
			log_level = std::max(0, log_level - 1);
			break;
		case 'h':
			opt.help = true;
		case 0:
			auto lname = long_options[option_index].name;
			if (std::string("help") == lname) {
				opt.help = true;
			}
			if (std::string("broker-command-address") == lname) {
				opt.master_config.command_listener.address = optarg;
			}
			if (std::string("broker-command-topic") == lname) {
				opt.master_config.command_listener.topic = optarg;
			}
		}
	}

	if (getopt_error) {
		LOG(5, "ERROR parsing command line options");
		opt.help = true;
		return 1;
	}

	printf("ess-file-writer-0.0.1  (ESS, BrightnESS)\n");
	printf("  %.7s\n", GIT_COMMIT);
	printf("  Contact: dominik.werder@psi.ch\n\n");

	if (opt.help) {
		printf("Forwards EPICS process variables to Kafka topics.\n"
		       "Controlled via JSON packets sent over the configuration topic.\n"
		       "\n"
		       "\n"
		       "forward-epics-to-kafka\n"
		       "  --help, -h\n"
		       "\n"
		       "  --test\n"
		       "      Run tests\n"
		       "\n"
		       "  --broker-configuration-address    host:port,host:port,...\n"
		       "      Kafka brokers to connect with for configuration updates.\n"
		       "      Default: %s\n"
		       "\n",
			opt.master_config.command_listener.address.c_str());

		printf("  --broker-configuration-topic      <topic-name>\n"
		       "      Topic name to listen to for configuration updates.\n"
		       "      Default: %s\n"
		       "\n",
			opt.master_config.command_listener.topic.c_str());

		printf("  --verbose\n"
		       "\n");
		return 1;
	}

	return 0;
}


#if HAVE_GTEST

TEST(config, read_simple) {
	return;
	LOG(3, "Test a simple configuration");
	using namespace BrightnESS::FileWriter;
	// TODO
	// * Input a predefined configuration message to setup a simple stream writing
	// * Connect outputs to test buffers
	// * Input a predefined message (or more) and test if it arrives at the correct ends
	MasterConfig conf_m;
	conf_m.test_mockup_command_listener = true;
	Master m(conf_m);
	ASSERT_NO_THROW( m.run() );
}

TEST(setup_with_kafka, setup_01) {
	using namespace BrightnESS::FileWriter;
	MasterConfig conf_m;
	MasterConfig conf_m2;
	std::thread t1;

	if (0) {
		t1 = std::thread( [] {
		});
		t1.join();
	}

	if (1) {
		auto cb = [conf_m2](){
			LOG(3, "on_consumer_connected");
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			TestCommandProducer tcp;
			auto aaa = conf_m2;
			//tcp.produce_simple_01(conf_m.command_listener);
		};
		Master m(conf_m);
		cb();
		m.on_consumer_connected(cb);
		ASSERT_NO_THROW( m.run() );
	}
}

#endif
