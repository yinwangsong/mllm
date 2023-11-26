//
// Created by 咸的鱼 on 2023/11/26.
//

#include "CPUReLU2.hpp"

namespace mllm {

CPUReLU2::CPUReLU2(Backend *bn, string opName, bool multiThread):support_multi_thread_(multiThread), Op(bn, std::move(opName)) {
}
ErrorCode CPUReLU2::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    return Op::reshape(inputs, outputs);
}
ErrorCode CPUReLU2::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    int batch = input->batch();
    int head = input->head();
    int seq = input->sequence();
    int dim = input->dimension();
#pragma omp parallel for collapse(4)
    for (int b = 0; b <batch ; ++b) {
        for (int h = 0; h < head; ++h) {
            for (int s = 0; s < seq; ++s) {
                for (int d = 0; d < dim; ++d) {
                    float value = input->dataAt<float>(b, h, s, d);
                    if (value < 0) {
                        value = 0;
                    }
                    //Square
                    value = std::pow(value, 2);
                    output->setDataAt<float>(b, h, s, d, value);
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}
ErrorCode CPUReLU2::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm