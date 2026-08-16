#include "core/graph.h"
#include "core/blob.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <vector>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        // 踩坑（详见 docs/作业完成说明.md 作业八）：
        //   - 执行顺序：必须先做规则 1（消除互逆 transpose 对），再做规则 2
        //     （融合进 matmul），否则本应被抵消的 transpose 会被错误融合成 transA；
        //   - 删除算子/重连输入时，必须同步维护双向关系（tensor 的 source/targets
        //     与 op 的 predecessors/successors），否则会残留失效的 weak_ptr，
        //     打印图时崩溃（段错误 / bad_weak_ptr）；
        //   - 两个 permute 互逆 ⇔ 复合为恒等置换：p2[p1[i]] == i；
        //   - 只交换最后两维 ⇔ 前 rank-2 维不动，最后两维互换；
        //   - 删除中间 tensor 前检查它是否还被其他算子消费（targets 非空则保留）。
        auto isInverse = [](const std::vector<int> &p1, const std::vector<int> &p2) {
            if(p1.size() != p2.size()) {
                return false;
            }
            for(size_t i = 0; i < p1.size(); i++) {
                if(p2[(size_t)p1[i]] != (int)i) {
                    return false;
                }
            }
            return true;
        };
        auto isSwapLastTwo = [](const std::vector<int> &perm) {
            size_t r = perm.size();
            if (r < 2)
                return false;
            for (size_t i = 0; i < r; ++i)
            {
                int expect = (int)i;             // 默认保持原位
                if (i == r - 2)                  // 倒数第二维映射到最后一位
                    expect = (int)(r - 1);
                else if (i == r - 1)             // 最后一位映射到倒数第二维
                    expect = (int)(r - 2);
                if (perm[i] != expect)
                    return false;
            }
            return true;
        };
        // ==================== 规则 1：去除冗余的互逆 transpose 对 ====================
        bool changed = true;
        while (changed){
            changed = false;
            for (auto &op : ops){
                if (op->getOpType() != OpType::Transpose) {
                    continue;
                }
                auto trans = as<TransposeObj>(op); // mid -> out
                auto mid = trans->getInputs(0);
                auto pred = mid->getSource();      // input -> mid 获取mid的生产者
                if (!pred || pred->getOpType() != OpType::Transpose){
                    continue;
                }
                // 安全条件：mid 只被 trans 这一个算子消费，否则不能删除 pred
                if (mid->getTargets().size() != 1) {
                    continue;
                }
                // 如果两个permute不互为逆置换，continue
                if (!isInverse(as<TransposeObj>(pred)->getPermute(),
                                trans->getPermute())) {
                    continue;
                }
    
                auto input = pred->getInputs(0);
                auto out = trans->getOutput(0);
                // 先记住 out 是否有消费者（决定它是否还能删除）
                bool outHasConsumers = !out->getTargets().empty();
                 // 把 out 的所有消费者（后继算子）的输入从 out 换成 input
                for (auto &succ : out->getTargets()) {
                    succ->replaceInput(out, input);
                    out->removeTarget(succ);
                    input->addTarget(succ);
                    succ->removePredecessors(trans);
                    trans->removeSuccessors(succ);
                }
                input->removeTarget(pred);
                mid->removeTarget(trans);
                trans->removePredecessors(pred);
                pred->removeSuccessors(trans);

                removeOperator(trans);
                removeOperator(pred);
                removeTensor(mid);

                // out 原本有消费者且已全部改接到 input，可以一并删除；
                // 若 out 是图的输出（本来就没有消费者），则予以保留。
                if (outHasConsumers) {
                    removeTensor(out);
                }
                changed = true;
                break; // 重新扫描
            }
        }
        // ============ 规则 2：把输入上的 transpose 融合进 matmul 的 trans 属性 ============
        for (auto &op : ops)
        {
            if (op->getOpType() != OpType::MatMul) {
                continue;
            }
            auto matmul = as<MatmulObj>(op);
            for (int i = 0; i < 2; ++i)
            {
                auto in = matmul->getInputs(i);
                auto src = in->getSource();
                if (!src || src->getOpType() != OpType::Transpose) {
                    continue;
                }
                // 安全条件：该 transpose 只被 matmul 一个算子消费
                if (in->getTargets().size() != 1) {
                    continue;
                }
                if (!isSwapLastTwo(as<TransposeObj>(src)->getPermute())) {
                    continue;
                }
                auto trans = as<TransposeObj>(src);
                auto newIn = trans->getInputs(0);
                if (i == 0) {
                    matmul->setTransA(true);
                }
                else{
                    matmul->setTransB(true);
                }
                matmul->replaceInput(in, newIn);
                in->removeTarget(matmul);
                newIn->addTarget(matmul);
                newIn->removeTarget(trans);
                matmul->removePredecessors(trans);
                trans->removeSuccessors(matmul);
                in->setSource(nullptr);
                removeOperator(trans);
                removeTensor(in);
            }
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        // 踩坑（详见 docs/作业完成说明.md 作业一 step3）：
        //   - 必须分两阶段：先对每个 tensor 做"模拟"分配（alloc/free，只算偏移量），
        //     最后统一调用 allocator.getPtr() 拿真实内存基址再绑定 Blob。
        //     不能边 alloc 边 getPtr()（alloc 开头有 IT_ASSERT(ptr == nullptr)）；
        //   - 建议用引用计数做内存复用：记录每个 tensor 剩余消费者个数（targets 数），
        //     某个 op 规划完后其输入计数减一，归零即 free，供后面算子复用；
        //   - 每个 tensor 只分配一次（用一个集合记录已分配的 tensor，避免重复分配）；
        //   - 重要：算子的"输出"必须先于"输入回收"来分配，
        //     否则输出会复用本算子输入的内存，而 kernel 执行时边读输入边写输出，
        //     两者重叠会导致结果错误；
        //   - 绑定方式：make_ref<BlobObj>(runtime, (char *)base + offset)。
        std::map<Tensor, size_t> refCount; // 每个 Tensor 的消费者总数
        std::map<Tensor, size_t> offsetMap;   // 每个 Tensor 的偏移
        std::unordered_set<Tensor> allocated; // 已分配的 Tensor

        for(auto &t: tensors) {
            refCount[t] = t->getTargets().size();
        }
         // 阶段一：模拟分配，规划每个 tensor 的偏移量
         for(auto& op : ops) {
             // 为算子的输入分配内存（每个 tensor 只规划一次）
             for(auto& t : op->getInputs()) {
                 if(allocated.find(t) == allocated.end()) {
                     offsetMap[t] = allocator.alloc(t->getBytes());
                     allocated.insert(t);
                 }
             }

             // 为算子的输出分配内存（先于输入回收，避免与输入重叠）
             for(auto& t : op->getOutputs()) {
                  if(allocated.find(t) == allocated.end()) {
                      offsetMap[t] = allocator.alloc(t->getBytes());
                      allocated.insert(t);
                  }
             }
            // 本算子的输入在本算子执行完后不再被它使用：
            // 引用计数减一，归零的输入（最后一个消费者已完成）立即回收，
            // 供后面的算子复用，从而降低峰值内存。
            for(auto& t : op->getInputs()) {
                --refCount[t];
                if(refCount[t] == 0) {
                    allocator.free(offsetMap[t], t->getBytes());
                }
            }
         }
         // 阶段二：真正分配内存，并按偏移量给每个 tensor 绑定 Blob
         auto base = allocator.getPtr();
         for(auto &[t, offset] : offsetMap) {
             t->setDataBlob(make_ref<BlobObj>(runtime, (char*)base + offset));
         }
         
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini