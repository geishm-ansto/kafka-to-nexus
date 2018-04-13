#include <gtest/gtest.h>
#include <FlatbufferReader.h>


namespace FileWriter {
  
  using ReaderPtr = FlatbufferReaderRegistry::ReaderPtr;
  
  class ReaderRegistrationTest : public ::testing::Test {
  public:
    void SetUp() override {
      std::map<std::string, ReaderPtr> &Readers = FlatbufferReaderRegistry::getReaders();
      Readers.clear();
    };
    
    void TearDown() override {
      
    };
  };
  
  class DumyReader : public FileWriter::FlatbufferReader {
  public:
    bool verify(Msg const &msg) const override {
      return true;
    }
    std::string source_name(Msg const &msg) const override {
      return std::string();
    }
    std::uint64_t timestamp(Msg const &msg) const override {
      return 0;
    }
  };
  
  TEST_F(ReaderRegistrationTest, SimpleRegistration) {
    std::map<std::string, ReaderPtr> &Readers = FlatbufferReaderRegistry::getReaders();
    std::string TestKey("temp");
    EXPECT_EQ(Readers.size(), 0);
    {
    FlatbufferReaderRegistry::Registrar<DumyReader> RegisterIt(TestKey);
    }
    EXPECT_EQ(Readers.size(), 1);
    EXPECT_NE(Readers.find(TestKey), Readers.end());
  }
  
  TEST_F(ReaderRegistrationTest, SameKeyRegistration) {
    std::string TestKey("temp");
    {
    FlatbufferReaderRegistry::Registrar<DumyReader> RegisterIt(TestKey);
    }
    EXPECT_THROW(FlatbufferReaderRegistry::Registrar<DumyReader> RegisterIt(TestKey), std::runtime_error);
  }
  
  TEST_F(ReaderRegistrationTest, KeyToShort) {
    std::string TestKey("tem");
    EXPECT_THROW(FlatbufferReaderRegistry::Registrar<DumyReader> RegisterIt(TestKey), std::runtime_error);
  }
  
  TEST_F(ReaderRegistrationTest, KeyToLong) {
    std::string TestKey("tempp");
    EXPECT_THROW(FlatbufferReaderRegistry::Registrar<DumyReader> RegisterIt(TestKey), std::runtime_error);
  }
  
  TEST_F(ReaderRegistrationTest, StrKeyFound) {
    std::string TestKey("t3mp");
    {
    FlatbufferReaderRegistry::Registrar<DumyReader> RegisterIt(TestKey);
    }
    EXPECT_NE(FlatbufferReaderRegistry::find(TestKey).get(), nullptr);
  }
  
  TEST_F(ReaderRegistrationTest, StrKeyNotFound) {
    std::string TestKey("t3mp");
    {
    FlatbufferReaderRegistry::Registrar<DumyReader> RegisterIt(TestKey);
    }
    std::string FailKey("trump");
    EXPECT_EQ(FlatbufferReaderRegistry::find(FailKey).get(), nullptr);
  }
  
  TEST_F(ReaderRegistrationTest, MsgKeyFound) {
    std::string TestKey("t3mp");
    std::string TestData("dumy" + TestKey + "data");
    {
    FlatbufferReaderRegistry::Registrar<DumyReader> RegisterIt(TestKey);
    }
    Msg TestMessage = Msg::owned(TestData.data(), TestData.size());
    EXPECT_NE(FlatbufferReaderRegistry::find(TestMessage).get(), nullptr);
  }
  
  TEST_F(ReaderRegistrationTest, MsgKeyNotFound) {
    std::string TestKey("t3mp");
    std::string FailKey("fail");
    std::string TestData("dumy" + FailKey + "data");
    {
    FlatbufferReaderRegistry::Registrar<DumyReader> RegisterIt(TestKey);
    }
    Msg TestMessage = Msg::owned(TestData.data(), TestData.size());
    EXPECT_EQ(FlatbufferReaderRegistry::find(TestMessage).get(), nullptr);
  }
  
  TEST_F(ReaderRegistrationTest, MsgShort) {
    std::string TestData("dumy");
    Msg TestMessage = Msg::owned(TestData.data(), TestData.size());
    EXPECT_EQ(FlatbufferReaderRegistry::find(TestMessage).get(), nullptr);
  }
} // namespace FileWriter
