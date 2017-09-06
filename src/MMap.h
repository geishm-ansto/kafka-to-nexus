#pragma once

#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

class MMap {
public:
  using ptr = std::unique_ptr<MMap>;
  using string = std::string;

  static ptr open(string fname, size_t size) {
    return create_inner(fname, size);
  }

  static ptr create(string fname, size_t size) {
    return create_inner(fname, size, true);
  }

  ~MMap() {
    if (munmap(shm_ptr, shm_size) != 0) {
      LOG(3, "munmap failed");
      exit(1);
    }
    shm_ptr = nullptr;
    if (::close(fd) != 0) {
      LOG(3, "could not close mmap file");
      exit(1);
    }
    fd = -1;
  }

  void *addr() const { return shm_ptr; }

  size_t size() const { return shm_size; }

private:
  int fd;
  void *shm_ptr;
  size_t shm_size = 0;
  MMap() {}
  static ptr create_inner(string fname, size_t size, bool create = false) {
    auto ret = ptr(new MMap);
    ret->fd = -1;
    ret->shm_ptr = nullptr;
    ret->shm_size = size;
    int flags = O_RDWR;
    if (create) {
      flags |= O_CREAT;
    }
    ret->fd = ::open(fname.data(), flags);
    if (ret->fd == -1) {
      LOG(3, "open failed");
      exit(1);
    }
    if (ftruncate(ret->fd, ret->shm_size) != 0) {
      LOG(3, "fail truncate");
      exit(1);
    }
    ret->shm_ptr = mmap64(nullptr, ret->shm_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, ret->fd, 0);
    if (sizeof(char *) != 8) {
      LOG(3, "just making sure");
      exit(1);
    }
    if (ret->shm_ptr == MAP_FAILED) {
      LOG(3, "mmap failed");
      exit(1);
    }
    LOG(3, "shm_ptr: {}", ret->shm_ptr);
    return ret;
  }
};
