#include "utils/operator_utils.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include <algorithm>

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    // - 从后往前对齐两个 shape（可用反向迭代器 rbegin/rend），
    //   较短的 shape 缺失的维度视为 1；
    // - 每维规则：两值相等、或其一为 1 时可广播，结果取较大者；
    //   否则断言失败；
    // - 因为是从后往前生成的，记得 std::reverse 回来
    //   （需要 #include <algorithm>）。
    Shape output_shape;
    auto itA = A.rbegin();
    auto itB = B.rbegin();
    while(itA != A.rend() || itB != B.rend()) {
        int dimA = (itA != A.rend()) ? *itA++ : 1; 
        int dimB = (itB != B.rend()) ? *itB++ : 1; 
        IT_ASSERT(dimA == dimB || dimA == 1 || dimB == 1,
            "Broadcast failed: " + vecToString(A) + " vs " + vecToString(B));
        output_shape.emplace_back(std::max(dimA, dimB));
    }
    std::reverse(output_shape.begin(), output_shape.end());
    return output_shape;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
