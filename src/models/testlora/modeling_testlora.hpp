//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef MODELING_TESTLORA_HPP
#define MODELING_TESTLORA_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_testlora.hpp"

using namespace mllm;


class testlora final : public Module {
    Layer up_proj;
    Layer down_proj;

public:
    explicit testlora(){
        int hidden_dim = 4096;
        int ffn_hidden = 8;
        up_proj = Linear(ffn_hidden, hidden_dim, false, "up");
        down_proj = Linear(ffn_hidden, hidden_dim, false, "down");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override  {


        std::cout<<123213;
        int in_dim = 4096;
        int out_dim = 11008;
        int r = 8;

        auto tensor_for_lora = inputs[0];
        auto lora_a = up_proj(inputs[1]);
        auto lora_b = down_proj(inputs[1]);
        lora_b = lora_b.transpose(SEQUENCE, DIMENSION);

        // tensor_for_lora = tensor_for_lora + tensor_for_lora;

        auto lora_merged = Tensor::mm(lora_a, lora_b);

        tensor_for_lora = tensor_for_lora - lora_merged;

        tensor_for_lora = tensor_for_lora + lora_merged;

        return {tensor_for_lora};
    }

};

#endif // MODELING_TESTLORA_HPP