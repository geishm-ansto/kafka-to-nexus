#pragma once

#include "logger.h"
#include <map>
#include <memory>
#include <mutex>

class Alloc {
public:
  using ptr = std::unique_ptr<Alloc>;
  using sptr = std::shared_ptr<Alloc>;

  static sptr create(void *base, void *ceil) {
    auto ret = sptr(new Alloc);
    return ret;
  }

  void stats() const {}

  static void tcache_flush() {}

  static void tcache_disable() {}

  void *alloc(size_t size) {
    auto addr = malloc(size);
    if (addr == nullptr) {
      LOG(Sev::Error, "fail alloc size: {}", size);
      exit(1);
    }
    return addr;
  }

  void use_this() {}

  void use_default() {}

  bool check_in_range(void *p) { return true; }

private:
  Alloc() {}
};