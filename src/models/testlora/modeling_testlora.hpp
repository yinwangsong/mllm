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

public:
    explicit testlora(){}
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override  {
        // auto tensor_for_lora = inputs[0];
        // auto lora_a = inputs[1];
        // auto lora_b = inputs[2];

        // auto lora_merged = Tensor::mm(lora_a, lora_b);

        // tensor_for_lora = tensor_for_lora - lora_merged;

        // tensor_for_lora = tensor_for_lora + lora_merged;

        // return {tensor_for_lora};
        auto x =  inputs[0];

        auto lora_a = inputs[1];
        auto lora_b = inputs[2];

        auto tmp = Tensor::mm(x, lora_a);
        auto out = Tensor::mm(tmp, lora_b);

        auto res = out + x;
        return {res};
    }

};

#endif // MODELING_TESTLORA_HPP