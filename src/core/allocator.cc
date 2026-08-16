#include "core/allocator.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        // - 推荐 first-fit：从低地址开始扫描 free list，找到第一个足够大的块；
        // - 块比需要的大时，把剩余部分切回 free list；
        // - 找不到合适块时，在 peak 处追加；若地址最大的空闲块恰好延伸到
        //   peak（first + second == peak），从它的起始地址向外扩展（见头文件踩坑）；
        // - 分配成功记得更新 used、peak（peak = max(peak, used)）。
        for(auto& it : freeBlocks) {
            if(it.second >= size) {
                size_t addr = it.first;
                if(it.second > size) {
                    freeBlocks[addr + size] = it.second - size;
                }
                freeBlocks.erase(it.first);
                used += size;
                peak = std::max(peak, used);
                return addr;
            }
        }
        // 如果地址最大的空闲块紧贴着内存末尾
        // 为了复用内存，就从该内存块开始向后拓展
        if(freeBlocks.empty() != true) {
            auto last = std::prev(freeBlocks.end());
            if(last->first + last->second == peak) {
                last->second += size;
                size_t addr = last->first;
                used += size;
                peak = addr + size;
                freeBlocks.erase(last);
                return addr;
            }
        }

        size_t addr = peak;
        used += size;
        peak += size;

        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        // - 把 [addr, addr + size) 放回 free list（used 相应减少）；
        // - 先与前一块合并（prev->first + prev->second == addr），
        //   再与后一块合并（it->first + it->second == next->first），顺序不能反。
        auto it = freeBlocks.emplace(addr, size).first;
        used -= size;

        if(it != freeBlocks.begin()) {
            auto prev = std::prev(it);
            if(prev->first + prev->second == addr) {
                prev->second += size;
                freeBlocks.erase(it);
                it = prev;
            }
        }
        auto next = std::next(it);
        if(next != freeBlocks.end() && it->first + it->second == next->first) {
            it->second += next->second;
            freeBlocks.erase(next);
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
