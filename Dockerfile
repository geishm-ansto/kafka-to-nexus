FROM screamingudder/ubuntu18.04-build-node:3.3.0 AS buildstage

ENV DEBIAN_FRONTEND=noninteractive

ARG http_proxy

ARG https_proxy

ARG local_conan_server

# Add local Conan server if one is defined in the environment
RUN if [ ! -z "$local_conan_server" ]; then conan remote add --insert 0 ess-dmsc-local "$local_conan_server"; fi

# Do the Conan install step first so that we don't have to rebuild all dependencies if something changes in the kafka_to_nexus source
RUN mkdir kafka_to_nexus
RUN mkdir kafka_to_nexus_src
COPY conan/conanfile.txt kafka_to_nexus_src/
RUN cd kafka_to_nexus && conan install --build=outdated ../kafka_to_nexus_src/conanfile.txt
COPY cmake kafka_to_nexus_src/cmake
COPY src kafka_to_nexus_src/src
COPY CMakeLists.txt Doxygen.conf kafka_to_nexus_src/

RUN cd kafka_to_nexus && \
    cmake -DCONAN="MANUAL" ../kafka_to_nexus_src && \
    make -j4 kafka-to-nexus

FROM ubuntu:18.04

RUN mkdir kafka_to_nexus
COPY --from=buildstage /home/jenkins/kafka_to_nexus/bin/kafka-to-nexus kafka_to_nexus/bin/
COPY --from=buildstage /home/jenkins/kafka_to_nexus/lib/ kafka_to_nexus/lib/
ENV LD_LIBRARY_PATH=/kafka_to_nexus/lib:$LD_LIBRARY_PATH

COPY docker_launch.sh .
CMD ["./docker_launch.sh"]
