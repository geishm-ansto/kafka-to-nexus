[![Build Status](https://jenkins.esss.dk/dm/job/ess-dmsc/job/kafka-to-nexus/job/master/badge/icon)](https://jenkins.esss.dk/dm/job/ess-dmsc/job/kafka-to-nexus/job/master/)
[![codecov](https://codecov.io/gh/ess-dmsc/kafka-to-nexus/branch/master/graph/badge.svg)](https://codecov.io/gh/ess-dmsc/kafka-to-nexus)
[![DOI](https://zenodo.org/badge/81435658.svg)](https://zenodo.org/badge/latestdoi/81435658)


# Kafka to Nexus File-Writer

Writes NeXus files from experiment data streamed through Apache Kafka.
Part of the ESS data streaming pipeline.

## Usage

```
  -h,--help                   Print this help message and exit
  --commands-json TEXT        Specify a json file to set config
  --command-uri URI (REQUIRED)
                              <//host[:port][/topic]> Kafka broker/topic to listen for commands
  --status-uri URI            <//host[:port][/topic]> Kafka broker/topic to publish status updates on
  --kafka-gelf TEXT           <//host[:port]/topic> Log to Graylog via Kafka GELF adapter
  --graylog-logger-address TEXT
                              <host:port> Log to Graylog via graylog_logger library
  -v,--verbosity INT=3        Set logging level. 3 == Error, 7 == Debug. Default: 3 (Error)
  --hdf-output-prefix TEXT    <absolute/or/relative/directory> Directory which gets prepended to the HDF output filenames in the file write commands
  --logpid-sleep              
  --use-signal-handler        
  --log-file TEXT             Specify file to log to
  --teamid UINT               
  --service-id TEXT           Identifier string for this filewriter instance. Otherwise by default a string containing hostname and process id.
  --status-master-interval UINT=2000
                              Interval in milliseconds for status updates
  --list_modules              List registered read and writer parts of file-writing modules and then exit.
  --streamer-ms-before-start  Streamer option - milliseconds before start time
  --streamer-ms-after-stop    Streamer option - milliseconds after stop time
  --streamer-start-time       Streamer option - start timestamp (milliseconds)
  --streamer-stop-time        Streamer option - stop timestamp (milliseconds)
  --stream-master-topic-write-interval
                              Stream-master option - topic write interval (milliseconds)
  -S,--kafka-config KEY VALUE ...
                              LibRDKafka options
  -c,--config-file TEXT       Read configuration from an ini file
```

### Configuration Files

The file-writer can be configured from a file via `--config-file <ini>` which mirrors the command line options.

For example:

```ini
command-uri=//broker[:port]/command-topic
status-uri=//broker[:port]/status-topic
commands-json=./commands.json
hdf-output-prefix=./absolute/or/relative/path/to/hdf/output/directory
service-id=this_is_filewriter_instance_HOST_PID_EXAMPLENAME
streamer-ms-before-start=123456
kafka-config=consumer.timeout.ms 501 fetch.message.max.bytes 1234 api.version.request true
```

Note: the Kafka options are key-value pairs and the file-writer can be given multiple by appending the key-value pair to the end of the command line option.

### Sending commands to the file-writer

Beyond the configuration options given at start-up, the file-writer can be sent commands via Kafka to control the actual file writing.

See the [commands](link goes here) for more information.

## Installation

The supported method for [installation](#install-via-conan) is via Conan.

### Prerequisites

The following minimum software is required to get started:

- Conan
- CMake >= 3.1.0
- Git
- A C++14 compatible compiler (preferably GCC or Clang)
- Doxygen (only required if you would like to generate the documentation)

Conan will install all the other required packages.

### Add the Conan remote repositories

Add the required remote repositories like so:

```bash
conan remote add conancommunity https://api.bintray.com/conan/conan-community/conan
conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan
conan remote add conan-transit https://api.bintray.com/conan/conan/conan-transit
conan remote add ess-dmsc https://api.bintray.com/conan/ess-dmsc/conan
```

### Build

From within the file-writer's top directory:

```bash
mkdir _build
cd _build
conan install ../conan --build=missing
cmake .. -DREQUIRE_GTEST=TRUE
make
```

To, optionally, generate the Doxygen documentation run:
```bash
make docs
```

### Running the unit tests

From the build directory:

```bash
./tests/UnitTests
```

### System tests

The system tests run a series of higher level tests interacting with a containerised instance of Kakfa and a data producer.

See [System Tests page](system-tests/README.md) for more information.

## Documentation

See the `docs` directory.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for information on submitting pull requests to this project.

## License

This project is licensed under the BSD 2-Clause "Simplified" License - see the [LICENSE.md](LICENSE.md) file for details.
