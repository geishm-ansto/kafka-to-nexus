[![Build Status](https://jenkins.esss.dk/dm/job/ess-dmsc/job/kafka-to-nexus/job/master/badge/icon)](https://jenkins.esss.dk/dm/job/ess-dmsc/job/kafka-to-nexus/job/master/)
[![codecov](https://codecov.io/gh/ess-dmsc/kafka-to-nexus/branch/master/graph/badge.svg)](https://codecov.io/gh/ess-dmsc/kafka-to-nexus)
[![DOI](https://zenodo.org/badge/81435658.svg)](https://zenodo.org/badge/latestdoi/81435658)


# Kafka to Nexus file writing

- [Usage](#usage)
- [Installation](#installation)
- [Configuration Files](#configuration-files)
- [Commands](#send-command-to-kafka-to-nexus)
- [Flatbuffer Schema Plugins](#flatbuffer-schema-plugins)
  - [f142 LogData](docs/docs/writer_module_f142_log_data.md)
  - [hs00 EventHistogram](docs/writer_module_hs00_event_histogram.md)


## Features

- Wait for file writing command from a Kafka topic
- Write data to file
- Writer plugins can be [configured via the json command](#writer-modules)


## Usage

```
./kafka-to-nexus --help
```

### Configuration Files

The file writer can be configured via `--config-file <ini>` which should contain command line options. 

#### Example configuration file:

```ini
command-uri=//broker[:port]/command-topic
status-uri=//broker[:port]/status-topic
commands-json=./commands.json
hdf-output-prefix=./absolute/or/relative/path/to/hdf/output/directory
service-id=this_is_filewriter_instance_HOST_PID_EXAMPLENAME
streamer-ms-before-start=123456
kafka-config=consumer.timeout.ms 501 fetch.message.max.bytes 1234 api.version.request true
```

Note that the kafka options are key value pairs and the filewriter can be given multiple by appending the key value pair to the end of the command line option. 

### Send command to kafka-to-nexus

Commands in the form of JSON messages are used to start and stop file writing.
Commands can be send through Kafka via the broker/topic specified by the
`--command-uri` option.  Commands can also be given in the configuration
file specified by `--config-file <file.json>` (see [commands in config file](#commands-can-be-given-in-the-configuration-file-as-well)).

In the command, the `nexus_structure` defines the HDF hierarchy.
The `nexus_structure` represents the HDF root object.  The following example
shows how the HDF tree can be constructed using `children`.
A child of type `stream` is a marker that some `HDFWriterModule` will insert
the data from a Kafka stream at that point in the hierarchy.  The options under
the key `stream` specify the details, at least the topic, the source name and
the `HDFWriterModule` which should be used for writing.
Depending on the `HDFWriterModule`, there will be more options specific to the
`HDFWriterModule`.

#### Command to start writing a file:

Further documentation:
- [Groups](docs/groups.md)
- [~~Datasets~~ documentation not yet written]()
- [Attributes](docs/attributes.md)
- [~~File Attributes~~ documentation not yet written]()
- [~~Streams~~ documentation not yet written]()

```json
{
  "nexus_structure": {
    "children": [
      {
        "type": "group",
        "name": "for_example_motor_0000",
        "attributes": {
          "NX_class": "NXinstrument"
        },
        "children": [
          {
            "type": "stream",
            "attributes": {
              "this_will_be_a_double": 0.123,
              "this_will_be_a_int64": 123
            },
            "stream": {
              "topic": "topic.with.multiple.sources",
              "source": "for_example_motor",
              "writer_module": "f142",
              "type": "float",
              "array_size": 4
            }
          },
          {
            "type": "dataset",
            "name": "some_static_dataset",
            "values": 42.24,
            "attributes": {
              "units": "Kelvin"
            }
          },
          {
            "type": "dataset",
            "name": "some_more_explicit_static_dataset",
            "dataset": {
              "space": "simple",
              "type": "uint64",
              "size": ["unlimited", 5, 6]
            },
            "values": [[[0, 1, 2, 3, 4, 5], [...], ...], [...], ...]
          },
          {
            "type": "dataset",
            "name": "string_scalar",
            "dataset": {
              "type": "string"
            },
            "values": "the-scalar-string"
          },
          {
            "type": "dataset",
            "name": "string_3d",
            "dataset": {
              "type": "string",
              "size": ["unlimited", 3, 2]
            },
            "values": [
              [
                ["string_0_0_0", "string_0_0_1"],
                ["string_0_1_0", "string_0_1_1"],
                ["string_0_2_0", "string_0_2_1"]
              ],
              [
                ["string_1_0_0", "string_1_0_1"],
                ["string_1_1_0", "string_1_1_1"],
                ["string_1_2_0", "string_1_2_1"]
              ]
            ]
          },
          {
            "type": "dataset",
            "name": "string_fixed_length_1d",
            "dataset": {
              "type":"string",
              "string_size": 32,
              "size": ["unlimited"]
            },
            "values": ["the-scalar-string", "another-one"],
            "attributes": [
            {
              "name": "scalar_attribute",
              "values": 42
            },
            {
              "name": "vector_attribute",
              "values": [1,2,3],
              "type": "uint32"
            }
            ]
          }
        ]
      }
    ]
  },
  "file_attributes": {
    "file_name": "some.h5"
  },
  "cmd": "FileWriter_new",
  "job_id": "unique-identifier",
  "broker": "localhost:9092",
  "start_time": <[OPTIONAL] timestamp in milliseconds>,
  "stop_time": <[OPTIONAL] timestamp in milliseconds>,
  "service_id": "[OPTIONAL] the_name_of_the_instance_which_should_interpret_this_command"
}
```


#### Command to exit the file writer:

```json
{
  "cmd": "FileWriter_exit",
  "service_id": "[OPTIONAL] the_name_of_the_instance_which_should_interpret_this_command"
}
```

#### Command to stop a single file:

```json
{
  "cmd": "FileWriter_stop",
  "job_id": "job-unique-identifier",
  "stop_time" : <[OPTIONAL] timestamp-in-milliseconds>,
  "service_id": "[OPTIONAL] the_name_of_the_instance_which_should_interpret_this_command"
}
```

#### Commands can be given in the configuration file as well:

```json
{
  "commands": [
    { "some command": "as discussed above" }
  ]
}
```


### Options for the f142 writer module

- `type`: The data type contained in the flat buffer. Can be `int8` to `int64`,
  similar for `uint8`, `float` and `double`.
- `array_size`: The size of the array. Scalar if not specified or `0`.


### Single Writer, Multiple Reader support

The filewriter can use HDF's Single Writer Multiple Reader feature (SWMR).

To write, as well as to read HDF files which use the SWMR feature requires at
least HDF version 1.10.  This means that also tools like HDFView are required
to link against at least HDF 1.10 in order to open HDF files which were written
in SWMR mode.

Please note, the HDF documentation itself warns:

"The HDF5 file that is accessed by SWMR HDF5 applications must be located on a
file system that complies with the POSIX `write()` semantics."

Also:

"The writer is not allowed to modify or append to any data items containing
variable-size datatypes (including string and region references datatypes)."

To enable SWMR when writing a file, add to the `FileWriter_new` command:

```json
{
  "use_hdf_swmr": true
}
```



## Installation

### Requirements

- cmake (at least 2.8.11)
- git
- flatbuffers (headers and working `flatc`)
- librdkafka
- hdf5
- libfmt (e.g. `yum install fmt fmt-devel` or `brew install fmt`)
- `streaming-data-types` repository (clone e.g. so that both `kafka-to-nexus`
  and `streaming-data-types` are in the same directory)
- Optional `graylog_logger`

Tooling

- conan
- cmake (minimum tested is 2.8.11)
- C++ compiler with c++11 support
- Doxygen if you would like to `make docs`

### Conan

For downloading and configuring dependencies there are three options which can be set using the `CONAN` CMake parameter with one of the following values:
- `AUTO` - (default) conan is used to download and configure dependencies, this is done automatically by CMake.
conan is required to be installed and in the `path`. A non-default conan profile can be specified by setting `CONAN_PROFILE`.
- `MANUAL` - conan can be run manually to generate a `conanbuildinfo.cmake` file in the build directory.
- `DISABLE` - conan is disabled. CMake will try to find system installed libraries or paths can be specified manually.


If using conan, the following remote repositories are required to be configured:

- https://api.bintray.com/conan/ess-dmsc/conan
- https://api.bintray.com/conan/conan-community/conan
- https://api.bintray.com/conan/vthiery/conan-packages
- https://api.bintray.com/conan/bincrafters/public-conan

You can add them by running
```
conan remote add <local-name> <remote-url>
```
where `<local-name>` must be substituted by a locally unique name. Configured
remotes can be listed with `conan remote list`.

### Build

As usual `cmake`, `make`.
```
conan install <path-to-source>/conan --build=missing
cmake <path-to-source> [-DREQUIRE_GTEST=TRUE]
make
make docs  # optional
```

#### Usage of your custom builds of the dependencies

If you have dependencies in non-standard locations:
Locations of dependencies can be supplied via the standard
`CMAKE_INCLUDE_PATH` and `CMAKE_LIBRARY_PATH` variables.

- `flatbuffers` Headers plus `flatc`, therefore set `CMAKE_INCLUDE_PATH` and `CMAKE_PROGRAM_PATH`.

- `HDF5`

- `graylog_logger` Additionally, set `USE_GRAYLOG_LOGGER=1`
  - cmake will report if it is found

- `libfmt` Header/Source-only
  - we expect `fmt/[format.cc, format.h]`

- `Google Test` (optional) Easiest way: `git clone https://github.com/google/googletest.git`
  in parallel to this repository, or give the repository location in
  `CMAKE_INCLUDE_PATH` or in `GOOGLETEST_REPOSITORY_DIR`.
  Enable gtest usage by `REQUIRE_GTEST=1`

If you like full fine-grained control over the locations, you can of course set
the locations directly as the package-specific variables which can be looked up
in the `Find...` scripts under `./cmake/` for each package.


### Using Ansible

Install using the playbook:

```
ansible-playbook -i hosts kafka-to-nexus.yml
```

The filewriter can be installed using the ansible playbook defined in
`ansible`. The file `roles/kafka-to-nexus/defaults/main.yml` defines
the variables used during installation. The variables `<dep>_src` and
`<dep>_version` are the remote source and the required version of the
dependency, `<dep>` the install location. The sources and builds of
the dependencies are kept in `sources` and `builds`.

`filewriter_inc`, `filewriter_lib` and `filewriter_bin` defines
`CMAKE_INCLUDE_PATH`, `CMAKE_LIBRARY_PATH` and `CMAKE_PROGRAM_PATH`.

The default installation has the following structure
```
/opt/software/sources/<package1>-<version>
/opt/software/sources/<package2>-<version>
...
/opt/software/builds/<package1>-<version>
/opt/software/builds/<package2>-<version>
...
/opt/software/<package1>-<version>
/opt/software/<package2>-<version>
...
/opt/software/sources/filewriter-<version>
/opt/software/builds/filewriter-<version>
```


## Writer Modules

Writer modules for the various flatbuffer schemas give the filewriter the
ability to parse the flatbuffers and write them to HDF5.

The actual parsing of the different FlatBuffer schemas and conversion to HDF5 is
handled by modules which register themselves via the `FlatbufferReaderRegistry`
and `HDFWriterModuleRegistry`.  For an example, please search for `Registrar` in
`src/schemas/hs00/`.  Support for new schemas can be added in the same way.


### Module for f142 LogData

[Documentation](docs/writer_module_f142_log_data.md).


### Module for hs00 EventHistogram

[Documentation](docs/writer_module_hs00_event_histogram.md).


## Running Tests

Tests are built only when `gtest` is detected.  If detected, the `cmake` output
contains
```
-- Using Google Test: [ DISCOVERED_LOCATION_OF_GTEST ]
```
with the location where it has found `gtest`.

To enable tests for the ``Streamer`` the [librdkafka-fake](https://github.com/ess-dmsc/librdkafka-fake) implementation of librdkafka must be enabled with
```
-DFAKE_RDKAFKA=<path>
```

Start the `gtest` based test suite via:
```
./tests/tests
```

## System Tests

See [System Tests page](system-tests/README.md).


## Performance

- [Profiling ev42 HDF writer module](docs/profile-ev42.md)


## Documents:

[Nexus-for-ESS](https://confluence.esss.lu.se/display/DMSC/NeXus+for+ESS)

[file-writer-2016-10-28](https://confluence.esss.lu.se/download/attachments/48202445/BrightNeXus.pdf?version=1&modificationDate=1477659873237&api=v2)


## Archive

These documents are outdated regarding technical documentation, but linked here
for archival:

- [NeXusFileWriterDesign](docs/NeXusFileWriterDesign.md) (Mark).


## Graph of dependencies and data flow

 Very early draft so far:

![Flow](flow.svg)

A pictorial representation of the implementation is ![File Writer overall design](docs/FileWriter.jpg)

## Streamer

According to the design the Streamer connects to Kafka (other
sources to be implemented) and consumes a message in the specified topic. Some features:

* one Streamer per topic
* multiple Source per streamer
* initial timestamp is specified using ``set_start_time``
* connection to the Kafka broker is nonblocking. If the broker address is invalid returns an error
* Kafka::Config and streamer options can be optionally configured using ``kafka`` and ``streamer`` fields in the configuration file. ``kafka`` can contain any option that RdKafka accepts. `streamer` accetps:
  - `ms-before-start` milliseconds before the `start_time` to start writing from
    - `consumer-timeout-ms` the maximum time in milliseconds the consumer waits
      before return with error status
    - `metadata-retry` maxim number of retries to connect to specifies broker
      before return an error

## DemuxTopic
Mapped 1:1 with topics (and Streamers) drives the message to the correct Source. Derived from classes MessageProcessor and TimeDifferenceFromMessage. The former provides an interface for processing new messages (usually write on disk), the latter the interface process old messaged with the aim of find the first message sent after ECP ```start ```message.
The two corresponding methods are

* process_message
* time_difference_from_message

Both receive the message payload and size. Return values are ProcessMessageResult and TimeDifferenceFromMessage_DT.

## StreamMaster 

The StreamMaster receives the array of DemuxTopic from
FileWriterCommand and instantiates the Streamer array according to the
topics. Eventually retrieves the list of brokers from Kafka.

* `start_time` and `stop_time` can be used to set the timestamp of
the first and last event to be written (see Streamer options);
* upon a `stop` message the ``Master`` can stop the writing;
* if a `status-uri` is configured sends a (JSON formatted) status
  report on the corresponding topic;
* a global `status` flag report the status of
``StreamMaster``. Definitions are in
`Status::StreamMasterErrorCode` (the function `Err2Str`
converts the error code into a human readable string). 
* each topic is written continously for at most `topic-write-interval`. this value can be configured in the config file (default 1000ms)

