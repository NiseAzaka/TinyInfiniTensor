#include "operators/matmul.h"
#include "core/common.h"
#include "core/tensor.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        // 踩坑（详见 docs/作业完成说明.md 作业七）：
        //   - m/n/k 的取值最容易写反！记 A 转置前为 [..., m, k]，
        //     B 转置前为 [..., k, n]：
        //       transA=false：m=dimsA[rankA-2]，k=dimsA[rankA-1]
        //       transA=true ：m=dimsA[rankA-1]，k=dimsA[rankA-2]
        //       transB=false：n=dimsB[rankB-1]，k=dimsB[rankB-2]
        //       transB=true ：n=dimsB[rankB-2]，k=dimsB[rankB-1]
        //     （trans 时 m/k 的来源正好对调）
        //   - 校验两边的 k 相等；
        //   - 输出 = 批量维（A/B 去掉最后两维后做双向广播，复用
        //     utils/operator_utils.h 的 infer_broadcast）拼接 [m, n]；
        //   - m/n/k 是成员变量，这里顺便赋值，供 toString() 使用。
        const auto A = inputs[0];
        const auto B = inputs[1];
        auto dimsA = A->getDims();
        auto dimsB = B->getDims();
        auto rankA = A->getRank();
        auto rankB = B->getRank();

        m = transA ? dimsA[rankA - 1] : dimsA[rankA - 2];
        k = transA ? dimsA[rankA - 2] : dimsA[rankA - 1];
        n = transB ? dimsB[rankB - 2] : dimsB[rankB - 1];
        IT_ASSERT(k == (transB ? dimsB[rankB - 1] : dimsB[rankB - 2]),
             "Matmul: inner dimensions mismatch");
        
        Shape batchA(dimsA.begin(), dimsA.end() - 2);
        Shape batchB(dimsB.begin(), dimsB.end() - 2);
        auto batch = infer_broadcast(batchA, batchB);

        auto dimsC = batch;
        dimsC.emplace_back(m);
        dimsC.emplace_back(n);
        return {{dimsC}};
    }

} // namespace infini