#include "CommandListener.h"
#include <string>
#include <vector>
#include <map>
#include "logger.h"
#include "helper.h"
#include "Master_handler.h"
#include "kafka_util.h"
#include <cassert>
#include <sys/types.h>
#include <unistd.h>
#include <future>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>

#include <librdkafka/rdkafka.h>
#include <librdkafka/rdkafkacpp.h>


namespace BrightnESS {
namespace FileWriter {

using std::vector;
using std::string;




#define KERR(err) if (err != 0) { LOG(3, "Kafka error code: {}", err); }


PollStatus::~PollStatus() {
	reset();
}

PollStatus PollStatus::Ok() {
	PollStatus ret;
	ret.state = 0;
	return ret;
}

PollStatus PollStatus::Err() {
	PollStatus ret;
	ret.state = -1;
	return ret;
}

PollStatus PollStatus::make_CmdMsg(std::unique_ptr<CmdMsg> x) {
	PollStatus ret;
	ret.state = 1;
	ret.data = x.release();
	return ret;
}

PollStatus::PollStatus(PollStatus && x)
:	state(std::move(x.state)),
	data(std::move(x.data))
{
}

PollStatus & PollStatus::operator = (PollStatus && x) {
	reset();
	std::swap(state, x.state);
	std::swap(data, x.data);
	return *this;
}

void PollStatus::reset() {
	if (state == 1) {
		if (auto x = (CmdMsg*)data) {
			delete x;
		}
	}
	state = -1;
	data = nullptr;
}

PollStatus::PollStatus() {
}

bool PollStatus::is_Ok() {
	return state == 0;
}

bool PollStatus::is_Err() {
	return state == -1;
}

std::unique_ptr<CmdMsg> PollStatus::is_CmdMsg() {
	if (state == 1) {
		std::unique_ptr<CmdMsg> ret((CmdMsg*)data);
		data = nullptr;
		return ret;
	}
	return nullptr;
}




class Consumer {
public:
Consumer(BrokerOpt opt);
~Consumer();
void start();
void dump_current_subscription();
PollStatus poll();
std::function<void()> * on_rebalance_assign = nullptr;
std::function<void(rd_kafka_topic_partition_list_t * plist)> on_rebalance_start = nullptr;
private:
BrokerOpt opt;
int poll_timeout_ms = 10;
static void cb_log(rd_kafka_t const * rk, int level, char const * fac, char const * buf);
static int cb_stats(rd_kafka_t * rk, char * json, size_t json_size, void * opaque);
static void cb_error(rd_kafka_t * rk, int err_i, char const * reason, void * opaque);
static void cb_rebalance(rd_kafka_t * rk, rd_kafka_resp_err_t err, rd_kafka_topic_partition_list_t * plist, void * opaque);
static void cb_consume(rd_kafka_message_t * msg, void * opaque);
rd_kafka_t * rk = nullptr;
//rd_kafka_topic_t * rkt = nullptr;
rd_kafka_topic_partition_list_t * plist = nullptr;
};


Consumer::Consumer(BrokerOpt opt) : opt(opt), poll_timeout_ms(200) {
	start();
}


Consumer::~Consumer() {
	LOG(-1, "~Consumer()");
	if (rk) {
		// commit offsets?
		if (0) {
			LOG(-1, "rd_kafka_unsubscribe");
			rd_kafka_unsubscribe(rk);
		}
		if (0) {
			LOG(-1, "rd_kafka_poll");
			int n1 = rd_kafka_poll(rk, 100);
			LOG(-1, "  served {} reuests", n1);
		}
		if (1) {
			LOG(-1, "rd_kafka_consumer_close");
			rd_kafka_consumer_close(rk);
		}
		// rd_kafka_consume_stop(rd_kafka_topic_t *, partition)  therefore low-level API?
		if (1) {
			LOG(-1, "rd_kafka_destroy");
			rd_kafka_destroy(rk);
			rk = nullptr;
		}
	}
	if (plist) {
		rd_kafka_topic_partition_list_destroy(plist);
		plist = nullptr;
	}
}


void Consumer::cb_log(rd_kafka_t const * rk, int level, char const * fac, char const * buf) {
	LOG(level, "cb_log: {}  fac: {}", buf, fac);
}

// Called from the poll() thread
void Consumer::cb_error(rd_kafka_t * rk, int err_i, char const * reason, void * opaque) {
	// cast necessary because of Kafka API design
	rd_kafka_resp_err_t err = (rd_kafka_resp_err_t) err_i;
	LOG(7, "ERROR Kafka Config: {}, {}, {}, {}", err_i, rd_kafka_err2name(err), rd_kafka_err2str(err), reason);
	// Could do something with this opaque:
	//auto self = static_cast<Consumer*>(opaque);
}



int Consumer::cb_stats(rd_kafka_t * rk, char * json, size_t json_size, void * opaque) {
	LOG(3, "INFO stats_cb {}  {:.{}}", json_size, json, json_size);
	// TODO
	// What does Kafka want us to return from this callback?
	return 0;
}



static void print_partition_list(rd_kafka_topic_partition_list_t * plist) {
	for (int i1 = 0; i1 < plist->cnt; ++i1) {
		auto & x = plist->elems[i1];
		LOG(3, "   {}  {}  {}", x.topic, x.partition, x.offset);
	}
}


void Consumer::cb_rebalance(rd_kafka_t * rk, rd_kafka_resp_err_t err, rd_kafka_topic_partition_list_t * plist, void * opaque) {
	rd_kafka_resp_err_t err2;
	LOG(3, "Consumer::cb_rebalance");
	auto self = static_cast<Consumer*>(opaque);
	switch (err) {
	case RD_KAFKA_RESP_ERR__ASSIGN_PARTITIONS:
		LOG(3, "rebalance_cb assign:");
		if (auto & cb = self->on_rebalance_start) {
			cb(plist);
		}
		print_partition_list(plist);
		err2 = rd_kafka_assign(rk, plist);
		if (err2 != RD_KAFKA_RESP_ERR_NO_ERROR) {
			LOG(9, "rebalance error: {}  {}", rd_kafka_err2name(err2), rd_kafka_err2str(err2));
		}
		if (self->on_rebalance_assign) {
			(*self->on_rebalance_assign)();
		}
		break;
	case RD_KAFKA_RESP_ERR__REVOKE_PARTITIONS:
		LOG(3, "rebalance_cb revoke:");
		print_partition_list(plist);
		err2 = rd_kafka_assign(rk, NULL);
		if (err2 != RD_KAFKA_RESP_ERR_NO_ERROR) {
			LOG(9, "rebalance error: {}  {}", rd_kafka_err2name(err2), rd_kafka_err2str(err2));
		}
		/*
		LOG(3, "commit offsets");
		err2 = rd_kafka_commit(rk, plist, 0);
		if (err2 != RD_KAFKA_RESP_ERR_NO_ERROR) {
			LOG(9, "commit error: {}  {}", rd_kafka_err2name(err2), rd_kafka_err2str(err2));
		}
		*/
		break;
	default:
		LOG(3, "rebalance_cb failure and revoke: {}", rd_kafka_err2str(err));
		err2 = rd_kafka_assign(rk, NULL);
		if (err2 != RD_KAFKA_RESP_ERR_NO_ERROR) {
			LOG(9, "rebalance error: {}  {}", rd_kafka_err2name(err2), rd_kafka_err2str(err2));
		}
		break;
	}
}



void Consumer::cb_consume(rd_kafka_message_t * msg, void * opaque) {
	//auto const & consumer = static_cast<Consumer*>(opaque);
	LOG(3, "consume_cb");

	if (msg) {
		//auto topic_name = rd_kafka_topic_name(msg->rkt);
		//int partition = msg->partition;
		if (msg->err == RD_KAFKA_RESP_ERR_NO_ERROR) {
			LOG(3, "GOT MESSAGE  {}  {:.{}}", msg->offset, (char*)msg->payload, msg->len);
		}
		else if (msg->err == RD_KAFKA_RESP_ERR__PARTITION_EOF) {
			// Just an advisory.  msg contains which partition it is.
		}
		else if (msg->err == RD_KAFKA_RESP_ERR__ALL_BROKERS_DOWN) {
			LOG(3, "RD_KAFKA_RESP_ERR__ALL_BROKERS_DOWN");
			return;
		}
		else if (msg->err == RD_KAFKA_RESP_ERR__BAD_MSG) {
			LOG(3, "RD_KAFKA_RESP_ERR__BAD_MSG");
			throw std::runtime_error("RD_KAFKA_RESP_ERR__BAD_MSG");
		}
		else if (msg->err == RD_KAFKA_RESP_ERR__DESTROY) {
			LOG(3, "RD_KAFKA_RESP_ERR__DESTROY");
			// Broker will go away soon
			LOG(3, "WARNING broker will go away");
		}
		else {
			LOG(3, "ERROR unhandled msg error: {} {}", rd_kafka_err2name(msg->err), rd_kafka_err2str(msg->err));
			throw std::runtime_error("unhandled error");
		}
	}
}



void Consumer::start() {
	int err;
	// librdkafka API sometimes wants to write errors into a buffer:
	int const errstr_N = 512;
	char errstr[errstr_N];

	auto conf = rd_kafka_conf_new();

	std::map<std::string, int> conf_ints {
		{"statistics.interval.ms",                  600 * 1000},
		{"metadata.request.timeout.ms",               2 * 1000},
		{"socket.timeout.ms",                         2 * 1000},
		{"session.timeout.ms",                        2 * 1000},
		{"coordinator.query.interval.ms",             2 * 1000},
		{"heartbeat.interval.ms",                          500},
		/*

		{"message.max.bytes",                 23 * 1024 * 1024},
		{"fetch.message.max.bytes",           23 * 1024 * 1024},
		{"receive.message.max.bytes",   5*    23 * 1024 * 1024},
		{"queue.buffering.max.messages",              2 * 1024},
		{"queue.buffering.max.ms",                        2000},
		{"batch.num.messages",                      100 * 1000},
		{"socket.send.buffer.bytes",          23 * 1024 * 1024},
		{"socket.receive.buffer.bytes",       23 * 1024 * 1024},

		// Consumer
		//{"queued.min.messages", "1"},
		*/
	};
	std::map<std::string, std::string> conf_strings {
		{"group.id", fmt::format("some-group-id", getpid())},
	};

	for (auto & c : conf_ints) {
		if (RD_KAFKA_CONF_OK != rd_kafka_conf_set(conf, c.first.c_str(), fmt::format("{:d}", c.second).c_str(), errstr, errstr_N)) {
			LOG(7, "error setting config: {}", c.first.c_str());
		}
	}
	for (auto & c : conf_strings) {
		if (RD_KAFKA_CONF_OK != rd_kafka_conf_set(conf, c.first.c_str(), c.second.c_str(), errstr, errstr_N)) {
			LOG(7, "error setting config: {}", c.first.c_str());
		}
	}

	// TODO
	// Release this resource later:
	//rd_kafka_topic_conf_t * topic_conf = nullptr;
	auto topic_conf = rd_kafka_topic_conf_new();
	//rd_kafka_topic_conf_set(topic_conf, "produce.offset.report", "true", errstr, errstr_N);
	//rd_kafka_topic_conf_set(topic_conf, "message.timeout.ms", "2000", errstr, errstr_N);
	//rd_kafka_topic_conf_set(topic_conf, "offset.store.method", "broker", errstr, errstr_N);

	rd_kafka_conf_set_default_topic_conf(conf, topic_conf);

	rd_kafka_conf_set_log_cb(conf, Consumer::cb_log);
	rd_kafka_conf_set_error_cb(conf, Consumer::cb_error);
	rd_kafka_conf_set_stats_cb(conf, Consumer::cb_stats);
	rd_kafka_conf_set_rebalance_cb(conf, Consumer::cb_rebalance);
	rd_kafka_conf_set_consume_cb(conf, cb_consume);
	rd_kafka_conf_set_consume_cb(conf, nullptr);

	rd_kafka_conf_set_opaque(conf, this);

	rk = rd_kafka_new(RD_KAFKA_CONSUMER, conf, errstr, errstr_N);
	if (!rk) {
		LOG(7, "ERROR can not create kafka handle: {}", errstr);
		throw std::runtime_error("can not create Kafka handle");
	}

	LOG(3, "New Kafka consumer: {}", rd_kafka_name(rk));

	int const LOG_DEBUG = 7;
	rd_kafka_set_log_level(rk, LOG_DEBUG);

	if (rd_kafka_brokers_add(rk, opt.address.c_str()) == 0) {
		LOG(7, "ERROR could not add brokers");
		throw std::runtime_error("could not add brokers");
	}

	rd_kafka_poll_set_consumer(rk);

	int partition = RD_KAFKA_PARTITION_UA;
	plist = rd_kafka_topic_partition_list_new(1);
	rd_kafka_topic_partition_list_add(plist, opt.topic.c_str(), partition);
	//rd_kafka_topic_partition_list_set_offset(plist, opt.topic.c_str(), partition, RD_KAFKA_OFFSET_BEGINNING);

	err = rd_kafka_subscribe(rk, plist);
	KERR(err);
	if (err) {
		LOG(7, "ERROR could not subscribe");
		throw std::runtime_error("can not subscribe");
	}
}


void Consumer::dump_current_subscription() {
	// Dump current subscription:
	rd_kafka_topic_partition_list_t * l1 = nullptr;
	rd_kafka_subscription(rk, &l1);
	if (l1) {
		for (int i1 = 0; i1 < l1->cnt; ++i1) {
			LOG(1, "subscribed topics: {}  {}  off {}", l1->elems[i1].topic, rd_kafka_err2str(l1->elems[i1].err), l1->elems[i1].offset);
		}
		rd_kafka_topic_partition_list_destroy(l1);
	}
}





PollStatus Consumer::poll() {
	if (0) {
		dump_current_subscription();
	}

	if (0) {
		rd_kafka_dump(stdout, rk);
	}

	auto ret = PollStatus::Err();

	if (1) {
		{
			//LOG(3, "rd_kafka_consumer_poll");
			auto msg = rd_kafka_consumer_poll(rk, poll_timeout_ms);

			if (msg != nullptr) {
				//LOG(3, "while-loop rd_kafka_consumer_poll returned non-null");
				//auto topic_name = rd_kafka_topic_name(msg->rkt);
				//int partition = msg->partition;
				if (msg->err == RD_KAFKA_RESP_ERR_NO_ERROR) {
					LOG(0, "Consuming offset: {}  {:.{}}", msg->offset, (char*)msg->payload, msg->len);
					auto mk = new CmdMsg_K;
					auto p1 = (char*)msg->payload;
					std::copy(p1, p1 + msg->len, std::back_inserter(mk->_str));
					std::unique_ptr<CmdMsg> msg(mk);
					ret = PollStatus::make_CmdMsg(std::move(msg));
				}
				else if (msg->err == RD_KAFKA_RESP_ERR__PARTITION_EOF) {
					// Just an advisory.  msg contains which partition it is.
					LOG(0, "RD_KAFKA_RESP_ERR__PARTITION_EOF");
				}
				else if (msg->err == RD_KAFKA_RESP_ERR__ALL_BROKERS_DOWN) {
					LOG(3, "RD_KAFKA_RESP_ERR__ALL_BROKERS_DOWN");
				}
				else if (msg->err == RD_KAFKA_RESP_ERR__BAD_MSG) {
					LOG(9, "RD_KAFKA_RESP_ERR__BAD_MSG");
				}
				else if (msg->err == RD_KAFKA_RESP_ERR__DESTROY) {
					LOG(3, "RD_KAFKA_RESP_ERR__DESTROY");
					// Broker will go away soon
				}
				else {
					LOG(9, "ERROR unhandled msg error: {} {}", rd_kafka_err2name(msg->err), rd_kafka_err2str(msg->err));
				}
				rd_kafka_message_destroy(msg);
			}
			else {
				//LOG(9, "msg returned from rd_kafka_consumer_poll is nullptr which it should not!");
			}
		}
	}

	return ret;
}






class ConsumerCPP {
public:
ConsumerCPP(BrokerOpt opt);
void start();
void poll();
void print_subscribed();
private:
BrokerOpt opt;
std::unique_ptr<RdKafka::Conf> gconf;
std::unique_ptr<RdKafka::Conf> tconf;
std::unique_ptr<RdKafka::KafkaConsumer> kcons;
std::unique_ptr<RdKafka::Topic> topic;
int32_t partition = RdKafka::Topic::PARTITION_UA;
};





ConsumerCPP::ConsumerCPP(BrokerOpt opt) : opt(opt) {
}

void ConsumerCPP::start() {
	// C++ Kafka API version
	string errstr;
	gconf = decltype(gconf)(RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL));
	gconf->set("metadata.broker.list", opt.address, errstr);
	if (errstr.size() > 0) {
		// yeah, seriously....
		LOG(3, "errstr: {}", errstr);
		throw BrokerFailure(errstr);
		errstr.clear();
	}
	auto unique_group_id = fmt::format("{}", getpid());
	gconf->set("group.id", unique_group_id, errstr);
	tconf = decltype(tconf)(RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC));
	kcons = decltype(kcons)(RdKafka::KafkaConsumer::create(gconf.get(), errstr));
	if (not kcons) {
		LOG(3, "ERROR can not create the consumer {}", errstr);
		throw BrokerFailure(errstr);
	}

	/*
	topic = decltype(topic)(RdKafka::Topic::create(kcons.get(), config.topic, tconf.get(), errstr));
	if (not topic) {
		LOG(3, "ERROR can not create the topic: {}", errstr);
		throw BrokerFailure(errstr);
	}
	*/

	vector<string> topics = { opt.topic };
	auto err = kcons->subscribe(topics);
	if (err != RdKafka::ERR_NO_ERROR) {
		LOG(3, "ERROR can not subscribe with KafkaConsumer {}", errstr);
		throw BrokerFailure(errstr);
	}

	//print_subscribed();
}


void ConsumerCPP::poll() {
	// C++ API
	print_subscribed();
	// Currently, run command-listener single-threaded
	int timeout_ms = 100;
	//LOG(3, "polling");
	auto msg = kcons->consume(timeout_ms);
	auto err = msg->err();
	if (err == RdKafka::ERR_NO_ERROR) {
		LOG(3, "GOT MESSAGE");
	}
	else if (err == RdKafka::ERR__TIMED_OUT) {
		//LOG(9, "ERR__TIMED_OUT");
	}
	else if (err == RdKafka::ERR__PARTITION_EOF) {
		//LOG(9, "ERR__PARTITION_EOF");
		// when topic is empty, poll results most of the time in a timeout,
		// so how is this error code anymore useful than just a timeout?
	}
	else {
		LOG(9, "ERROR while polling for messages: {}", RdKafka::err2str(err));
	}
}



void ConsumerCPP::print_subscribed() {
	vector<RdKafka::TopicPartition*> topic_partitions;
	auto err = kcons->assignment(topic_partitions);
	string errstr;
	if (err != RdKafka::ERR_NO_ERROR) {
		LOG(3, "ERROR can not start Consumer {}", errstr);
		throw BrokerFailure(errstr);
	}
	LOG(3, "Currently subscribed to:");
	for (auto & tp : topic_partitions) {
		LOG(3, "Topic: {}  Partition: {}  Offset: {}", tp->topic(), tp->partition(), tp->offset());
	}
}









CommandListener::CommandListener(CommandListenerConfig config) : config(config) { }

CommandListener::~CommandListener() {
}

void CommandListener::start() {
	if (is_mockup) {
		LOG(1, "is_mockup, no Kafka init");
		return;
	}
	BrokerOpt opt;
	opt.address = config.address;
	opt.topic = config.topic;
	leg_consumer.reset(new Consumer(opt));
	leg_consumer->on_rebalance_assign = config.on_rebalance_assign;
	if (config.start_at_command_offset >= 0) {
		int n1 = config.start_at_command_offset;
		leg_consumer->on_rebalance_start = [n1] (rd_kafka_topic_partition_list_t * plist) {
			for (int i1 = 0; i1 < plist->cnt; ++i1) {
				plist->elems[i1].offset = n1;
			}
		};
	}
}


class ConsumeCallback : public RdKafka::ConsumeCb {
public:
void consume_cb(RdKafka::Message & msg, void * opaque) {
	switch (msg.err()) {
	case RdKafka::ERR__TIMED_OUT:
		break;

	case RdKafka::ERR_NO_ERROR:
		//msg.len();
		//msg.payload();
		//msg.offset();
		//msg.key();  can be nullptr
		break;

	case RdKafka::ERR__PARTITION_EOF:
		// Last message
		break;

	case RdKafka::ERR__UNKNOWN_TOPIC:
	case RdKafka::ERR__UNKNOWN_PARTITION:
		//msg.errstr()
		break;

	default:
		//msg.errstr()
		break;
	}
}
};


PollStatus CommandListener::poll() {
	if (leg_consumer) {
		return leg_consumer->poll();
	}
	return PollStatus::Err();
}





class Producer {
public:
Producer(BrokerOpt opt);
~Producer();
//void produce(void const * msg_data, int msg_size);
void poll_outq();
static void cb_delivered(rd_kafka_t * rk, rd_kafka_message_t const * msg, void * opaque);
static void cb_error(rd_kafka_t * rk, int err_i, char const * reason, void * opaque);
static int cb_stats(rd_kafka_t * rk, char * json, size_t json_len, void * opaque);
static void cb_log(rd_kafka_t const * rk, int level, char const * fac, char const * buf);
rd_kafka_t * rd_kafka_ptr() const;
std::function<void(rd_kafka_message_t const * msg)> * on_delivery = nullptr;
private:
BrokerOpt opt;
int poll_timeout_ms = 10;
rd_kafka_t * rk = nullptr;
//rd_kafka_topic_t * rkt = nullptr;
rd_kafka_topic_partition_list_t * plist = nullptr;
};



class ProducerTopic {
public:
ProducerTopic(Producer const & producer, string name);
~ProducerTopic();
void produce(void * msg_data, int msg_size);
private:
Producer const & producer;
rd_kafka_topic_t * rkt = nullptr;
string _name;
};






void Producer::cb_delivered(rd_kafka_t * rk, rd_kafka_message_t const * msg, void * opaque) {
	auto self = static_cast<Producer*>(opaque);
	if (!msg->err) {
		if (self->on_delivery) {
			(*self->on_delivery)(msg);
		}
		if (true) {
			LOG(0, "Ok delivered ({}, p {}, offset {}, len {}): {:.{}}\n",
				rd_kafka_name(rk),
				msg->partition, msg->offset, msg->len,
				(char const *)msg->payload, (int)msg->len
			);
		}
	}
	else {
		LOG(6, "ERROR on delivery, {}, topic {}, {} [{}] {}",
			rd_kafka_name(rk),
			rd_kafka_topic_name(msg->rkt),
			rd_kafka_err2name(msg->err),
			msg->err,
			rd_kafka_err2str(msg->err)
		);
	}
}


void Producer::cb_error(rd_kafka_t * rk, int err_i, char const * reason, void * opaque) {
	// cast necessary because of Kafka API design
	rd_kafka_resp_err_t err = (rd_kafka_resp_err_t) err_i;
	LOG(7, "ERROR {} {}, {}, {}, {}", rd_kafka_name(rk), err_i, rd_kafka_err2name(err), rd_kafka_err2str(err), reason);
}


int Producer::cb_stats(rd_kafka_t * rk, char * json, size_t json_len, void * opaque) {
	LOG(3, "INFO cb_stats {} length {}   {:.{}}", rd_kafka_name(rk), json_len, json, json_len);
	// What does Kafka want us to return from this callback?
	return 0;
}


void Producer::cb_log(rd_kafka_t const * rk, int level, char const * fac, char const * buf) {
	LOG(level, "{}  {}  fac: {}", rd_kafka_name(rk), buf, fac);
}








Producer::~Producer() {
	LOG(-1, "~Producer");
	if (rk) {
		int ns = 10;
		while (rd_kafka_outq_len(rk) > 0) {
			auto n1 = rd_kafka_poll(rk, ns);
			if (n1 > 0) {
				LOG(-1, "rd_kafka_poll handled {}, timeout {}", n1, ns);
			}
			ns = std::min(500, ns << 2);
		}
		LOG(-1, "rd_kafka_destroy");
		rd_kafka_destroy(rk);
		rk = nullptr;
	}
}


Producer::Producer(BrokerOpt opt) : opt(opt), poll_timeout_ms(100) {
	std::map<std::string, int> conf_ints {
		{"queue.buffering.max.ms",                         200},
		/*
		{"statistics.interval.ms",                   20 * 1000},
		{"metadata.request.timeout.ms",              15 * 1000},
		{"socket.timeout.ms",                         4 * 1000},
		{"session.timeout.ms",                       15 * 1000},

		{"message.max.bytes",                 23 * 1024 * 1024},
		//{"message.max.bytes",                       512 * 1024},

		// check again these two?
		{"fetch.message.max.bytes",            3 * 1024 * 1024},
		{"receive.message.max.bytes",          3 * 1024 * 1024},

		{"queue.buffering.max.messages",       2 * 1000 * 1000},
		//{"queue.buffering.max.kbytes",              800 * 1024},
		{"queue.buffering.max.ms",                        1000},

		// Total MessageSet size limited by message.max.bytes
		{"batch.num.messages",                      100 * 1000},
		{"socket.send.buffer.bytes",          23 * 1024 * 1024},
		{"socket.receive.buffer.bytes",       23 * 1024 * 1024},

		// Consumer
		//{"queued.min.messages", "1"},
		*/
	};

	// librdkafka API sometimes wants to write errors into a buffer:
	vector<char> errstr;
	errstr.resize(512);

	rd_kafka_conf_t * conf = 0;
	conf = rd_kafka_conf_new();
	rd_kafka_conf_set_dr_msg_cb(conf, Producer::cb_delivered);
	rd_kafka_conf_set_error_cb(conf, Producer::cb_error);
	rd_kafka_conf_set_stats_cb(conf, Producer::cb_stats);
	rd_kafka_conf_set_log_cb(conf, Producer::cb_log);

	rd_kafka_conf_set_opaque(conf, this);
	LOG(-1, "Producer opaque: {}", (void*)this);

	for (auto & c : conf_ints) {
		LOG(7, "Set config: {} = {}", c.first.c_str(), c.second);
		if (RD_KAFKA_CONF_OK != rd_kafka_conf_set(conf, c.first.c_str(), fmt::format("{:d}", c.second).c_str(), errstr.data(), errstr.size())) {
			LOG(7, "ERROR setting config: {}", c.first.c_str());
		}
	}

	rk = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr.data(), errstr.size());
	if (!rk) {
		LOG(7, "ERROR can not create kafka handle: {}", errstr.data());
		throw std::runtime_error("can not create Kafka handle");
	}

	rd_kafka_set_log_level(rk, 1);

	LOG(3, "New Kafka {} with brokers: {}", rd_kafka_name(rk), opt.address.c_str());
	if (rd_kafka_brokers_add(rk, opt.address.c_str()) == 0) {
		LOG(7, "ERROR could not add brokers");
		throw std::runtime_error("could not add brokers");
	}
}



void Producer::poll_outq() {
	while (rd_kafka_outq_len(rk) > 0) {
		rd_kafka_poll(rk, 50);
	}
}



rd_kafka_t * Producer::rd_kafka_ptr() const {
	return rk;
}




ProducerTopic::~ProducerTopic() {
	LOG(-1, "~ProducerTopic");
	if (rkt) {
		auto rk = producer.rd_kafka_ptr();
		int ns = 10;
		while (rd_kafka_outq_len(rk) > 0) {
			auto n1 = rd_kafka_poll(rk, ns);
			if (n1 > 0) {
				LOG(-1, "rd_kafka_poll handled {}, timeout {}", n1, ns);
			}
			ns = std::min(500, ns << 2);
		}
		LOG(-1, "rd_kafka_topic_destroy");
		rd_kafka_topic_destroy(rkt);
		rkt = nullptr;
	}
}


ProducerTopic::ProducerTopic(Producer const & producer, string name) : producer(producer), _name(name) {
	std::map<std::string, std::string> conf_strings {
		{"produce.offset.report",          "true"},
		/*
		{"request.required.acks", "0"},
		{"message.timeout.ms", "15000"},
		*/
	};
	vector<char> errstr(512);
	rd_kafka_topic_conf_t * topic_conf = rd_kafka_topic_conf_new();
	for (auto & c : conf_strings) {
		auto x = rd_kafka_topic_conf_set(topic_conf, c.first.c_str(), c.second.c_str(), errstr.data(), errstr.size());
		if (x != RD_KAFKA_CONF_OK) {
			LOG(7, "error setting config {}  {}", c.first.c_str(), errstr.data());
		}
	}

	// rd_kafka_msg_partitioner_random, rd_kafka_msg_partitioner_consistent, rd_kafka_msg_partitioner_consistent_random
	//rd_kafka_topic_conf_set_partitioner_cb(topic_conf, rd_kafka_msg_partitioner_random);

	rkt = rd_kafka_topic_new(producer.rd_kafka_ptr(), _name.c_str(), topic_conf);
	if (rkt == nullptr) {
		// Seems like Kafka uses the system error code?
		auto errstr = rd_kafka_err2str(rd_kafka_errno2err(errno));
		LOG(7, "ERROR could not create Kafka topic: {}", errstr);
		throw std::exception();
	}
	LOG(-1, "Ctor topic {} finished", rd_kafka_topic_name(rkt));
}


void ProducerTopic::produce(void * msg_data, int msg_size) {
	if (not rkt) {
		throw std::runtime_error("ERROR tried to produce on uninitialized rkt");
	}
	int x;
	int32_t partition = RD_KAFKA_PARTITION_UA;

	// Optional:
	void const * key = NULL;
	size_t key_len = 0;

	void * opaque = (void*) 794613;
	// no flags means that we reown our buffer when Kafka calls our callback.
	int msgflags = RD_KAFKA_MSG_F_COPY; // 0, RD_KAFKA_MSG_F_COPY, RD_KAFKA_MSG_F_FREE

	// TODO
	// How does Kafka report the error?
	// API docs state that error codes are given in 'errno'
	// Check that this is thread safe ?!?

	x = rd_kafka_produce(rkt, partition, msgflags, msg_data, msg_size, key, key_len, opaque);
	if (x == RD_KAFKA_RESP_ERR__QUEUE_FULL) {
		LOG(7, "ERROR OutQ: {}  QUEUE_FULL", rd_kafka_outq_len(producer.rd_kafka_ptr()));
		return;
	}
	if (x == RD_KAFKA_RESP_ERR_MSG_SIZE_TOO_LARGE) {
		LOG(7, "ERROR OutQ: {}  TOO_LARGE", rd_kafka_outq_len(producer.rd_kafka_ptr()));
		return;
	}
	if (x != 0) {
		LOG(7, "ERROR on produce topic {}  partition {}   {}",
			rd_kafka_topic_name(rkt),
			partition,
			rd_kafka_err2str(rd_kafka_last_error())
		);
		return;
	}
	LOG(-1, "sent to topic {} partition {}", rd_kafka_topic_name(rkt), partition);
}



int64_t TestCommandProducer::produce_simple_01(CommandListenerConfig config) {
	{
		BrokerOpt opt;
		opt.address = config.address;
		opt.topic = config.topic;
		Producer p(opt);
		std::promise<int64_t> offset;
		std::function<void(rd_kafka_message_t const * msg)> cb = [&offset](rd_kafka_message_t const * msg) {
			offset.set_value(msg->offset);
		};
		p.on_delivery = &cb;
		ProducerTopic pt(p, opt.topic);
		auto v1 = gulp("test/msg-conf-new-01.json");
		pt.produce(v1.data(), v1.size());
		p.poll_outq();
		auto fut = offset.get_future();
		auto x = fut.wait_for(std::chrono::milliseconds(2000));
		if (x != std::future_status::ready) {
			LOG(9, "Timeout on production of test message");
		}
		else {
			return fut.get();
		}
	}
	return -101;

	LOG(3, "Use for configuration the topic {}", config.topic);
	string errstr;
	auto gconf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
	gconf->set("metadata.broker.list", config.address, errstr);

	auto producer = RdKafka::Producer::create(gconf, errstr);
	auto topic = RdKafka::Topic::create(producer, config.topic, nullptr, errstr);
	if (errstr.size() > 0) {
		auto e = "ERROR can not create topic";
		LOG(9, e);
		throw BrokerFailure(e);
	}

	void * msg_data = nullptr;
	int msg_size = 0;

	auto v1 = gulp("test/msg-conf-new-01.json");
	msg_data = v1.data();
	msg_size = v1.size();

	RdKafka::ErrorCode err;
	err = producer->produce(topic, RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY, msg_data, msg_size, nullptr, nullptr);
	if (err != RdKafka::ERR_NO_ERROR) {
		auto e = "ERROR produce gave error";
		LOG(3, e);
		throw BrokerFailure(e);
	}
}


}
}



#if HAVE_GTEST
#include <gtest/gtest.h>

TEST(librdkafka, basics) {
	ASSERT_EQ(RD_KAFKA_RESP_ERR_NO_ERROR, 0);
}

#endif