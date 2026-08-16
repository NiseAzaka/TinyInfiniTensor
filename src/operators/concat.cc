#include "operators/concat.h"
#include "utils/operator_utils.h"
#include <cstddef>

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    // 除 dim 维外其它维度保持不变，dim 维大小为所有输入在该维大小之和；
    // 构造函数已把 dim 归一化到 [0, rank)，实现里再用 get_real_axis(dim, rank)
    // 防御一次负轴更稳妥；
    // 注意先 dims[dim] = 0 再累加，且所有输入的 rank 必须一致。
    // 踩坑：除 dim 维外，其它维度也必须逐一断言相等（ONNX 要求所有输入除
    // concat 维外形状相同），只检查 rank 是不够的。
    // 注意:上面 auto rank 推导出来的类型是 size_t 即unsigned long
    size_t concatDim = get_real_axis(dim, rank);
    dims[dim] = 0;
    for(auto& input : inputs) {
        IT_ASSERT(input->getRank() == rank);
        for (size_t i = 0; i < rank; ++i) {
            if (i != concatDim) {                // 除拼接维外，其余维度必须相同
                IT_ASSERT(input->getDims()[i] == dims[i],
                    "Concat: mismatch on dim " + std::to_string(i));
            }
        }
        dims[concatDim] += (input->getDims())[concatDim];
    }

    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
