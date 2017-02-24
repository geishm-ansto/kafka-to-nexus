#include "HDFFile.h"
#include "HDFFile_h5.h"
#include <array>
#include <chrono>
#include <ctime>
#include <hdf5.h>
#include "SchemaRegistry.h"
#include "logger.h"
#include <flatbuffers/flatbuffers.h>
#include <unistd.h>
#include "date/date.h"
#define HAS_REMOTE_API 0
#include "date/tz.h"


namespace BrightnESS {
namespace FileWriter {

class HDFFile_impl {
	friend class HDFFile;
	hid_t h5file = -1;
};

HDFFile::HDFFile() {
	// Keep this.  Will be used later to test against different lib versions
	#if H5_VERSION_GE(1, 8, 0) && H5_VERSION_LE(1, 10, 99)
		unsigned int maj, min, rel;
		H5get_libversion(&maj, &min, &rel);
	#else
		static_assert(false, "Unexpected HDF version");
	#endif

	impl.reset(new HDFFile_impl);
}

HDFFile::~HDFFile() {
	if (impl->h5file >= 0) {
		std::array<char, 512> fname;
		H5Fget_name(impl->h5file, fname.data(), fname.size());
		LOG(2, "flush file {}", fname.data());
		H5Fflush(impl->h5file, H5F_SCOPE_LOCAL);
		H5Fclose(impl->h5file);
	}
}

int HDFFile::init(std::string filename) {
	using std::string;
	using std::vector;
	auto x = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (x < 0) {
		std::array<char, 256> cwd;
		getcwd(cwd.data(), cwd.size());
		LOG(7, "ERROR could not create the HDF file: {}  cwd: {}", filename, cwd.data());
		return -1;
	}
	LOG(2, "INFO opened the file hid: {}", x);
	impl->h5file = x;

	auto f1 = x;

	auto lcpl = H5Pcreate(H5P_LINK_CREATE);
	H5Pset_char_encoding(lcpl, H5T_CSET_UTF8);
	auto acpl = H5Pcreate(H5P_ATTRIBUTE_CREATE);
	H5Pset_char_encoding(acpl, H5T_CSET_UTF8);
	auto strfix = H5Tcopy(H5T_C_S1);
	H5Tset_cset(strfix, H5T_CSET_UTF8);
	H5Tset_size(strfix, 1);
	auto dsp_sc = H5Screate(H5S_SCALAR);

	{
		auto payload = filename;
		H5Tset_size(strfix, payload.size());
		auto at = H5Acreate2(f1, "file_name", strfix, dsp_sc, acpl, H5P_DEFAULT);
		H5Awrite(at, strfix, payload.data());
		H5Aclose(at);
		//H5Eprint2(H5E_DEFAULT, nullptr);
	}

	{
		vector<char> s1(64);
		if (false) {
			// std way
			using namespace std::chrono;
			auto now = system_clock::now();
			auto time = system_clock::to_time_t(now);
			strftime(s1.data(), s1.size(), "%Y-%m-%dT%H:%M:%S%z", localtime(&time));
		}
		{
			// date way
			using namespace date;
			using namespace std::chrono;
			auto now2 = make_zoned(current_zone(), floor<milliseconds>(system_clock::now()));
			auto s2 = format("%Y-%m-%dT%H:%M:%S%z", now2);
			std::copy(s2.c_str(), s2.c_str() + s2.size(), s1.data());
		}
		H5Tset_size(strfix, s1.size());
		auto at = H5Acreate2(f1, "file_time", strfix, dsp_sc, acpl, H5P_DEFAULT);
		H5Awrite(at, strfix, s1.data());
		H5Aclose(at);
	}

	{
		// top level NXentry
		auto e = H5Gcreate2(f1, "entry", lcpl, H5P_DEFAULT, H5P_DEFAULT);
		string s1 {"NXentry"};
		H5Tset_size(strfix, s1.size());
		auto at = H5Acreate2(f1, "NX_class", strfix, dsp_sc, acpl, H5P_DEFAULT);
		H5Awrite(at, strfix, s1.data());
		H5Aclose(at);
	}

	H5Sclose(dsp_sc);
	H5Pclose(lcpl);
	H5Pclose(acpl);

	return 0;
}

void HDFFile::flush() {
	H5Fflush(impl->h5file, H5F_SCOPE_LOCAL);
}

HDFFile_h5 HDFFile::h5file_detail() {
	return HDFFile_h5(impl->h5file);
}

HDFFile_h5::HDFFile_h5(hid_t h5file) : _h5file(h5file) {
}

hid_t HDFFile_h5::h5file() {
	return _h5file;
}

std::unique_ptr<FBSchemaReader> FBSchemaReader::create(Msg msg) {
	static_assert(FLATBUFFERS_LITTLEENDIAN, "Requires currently little endian");
	if (msg.size < 8) {
		LOG(3, "ERROR message is too small");
		return nullptr;
	}
	Schemas::FBID fbid;
	memcpy(&fbid, msg.data + 4, 4);
	if (auto & cr = Schemas::SchemaRegistry::find(fbid)) {
		return cr->create_reader();
	}
	return nullptr;
}

std::unique_ptr<FBSchemaWriter> FBSchemaReader::create_writer() {
	return create_writer_impl();
}

FBSchemaReader::~FBSchemaReader() {
}

std::string FBSchemaReader::sourcename(Msg msg) {
	return sourcename_impl(msg);
}

uint64_t FBSchemaReader::ts(Msg msg) {
	return ts_impl(msg);
}

uint64_t FBSchemaReader::teamid(Msg & msg) {
	return teamid_impl(msg);
}

uint64_t FBSchemaReader::teamid_impl(Msg & msg) {
	return 0;
}


FBSchemaWriter::FBSchemaWriter() {
}

FBSchemaWriter::~FBSchemaWriter() {
}

void FBSchemaWriter::init(HDFFile * hdf_file, std::string const & sourcename, Msg msg) {
	this->hdf_file = hdf_file;
	init_impl(sourcename, msg);
}

WriteResult FBSchemaWriter::write(Msg msg) {
	return write_impl(msg);
}



}
}
