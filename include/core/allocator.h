#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {
  class Allocator
  {
  private:
    Runtime runtime;

    size_t used;

    size_t peak;

    size_t alignment;

    // pointer to the memory actually allocated
    void *ptr;

    // =================================== 作业 ===================================
    // TODO：可能需要设计一个数据结构来存储free block，以便于管理和合并
    // HINT: 可以使用一个 map 来存储 free block，key 为 block 的起始/结尾地址，value 为 block 的大小
    // =================================== 作业 ===================================
    // 踩坑（详见 docs/作业完成说明.md 作业一）：
    //   - alloc/free 只是"模拟"分配（函数开头有 IT_ASSERT(ptr == nullptr)），
    //     真实内存必须在全部规划完成之后通过 getPtr() 一次性分配；
    //   - alloc 需要支持"末尾空闲块扩展"（地址最大的空闲块延伸到 peak 时，
    //     可以从该块起始地址继续向外扩展），否则 test_allocator 的
    //     testAllocWithEndFreeBlock 无法通过；
    //   - free 时记得与前一块、后一块做合并（先向前、再向后），避免碎片。
    std::map<size_t, size_t> freeBlocks;
  public:
    Allocator(Runtime runtime);

    virtual ~Allocator();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    size_t alloc(size_t size);

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    void free(size_t addr, size_t size);

    // function: perform actual memory allocation
    // return: pointer to the head address of the allocated memory
    void *getPtr();

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    size_t getAlignedSize(size_t size);
  };
}
