//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#ifndef MODELING_TESTLORA_SIDEINFERENCE_HPP
#define MODELING_TESTLORA_SIDEINFERENCE_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_testlora_sideinference.hpp"

using namespace mllm;


class testlora_sideinference final : public Module {

public:
    explicit testlora_sideinference(){}
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override  {
        auto x =  inputs[0];

        auto lora_a = inputs[1];
        auto lora_b = inputs[2];

        auto tmp = Tensor::mm(x, lora_a);
        auto out = Tensor::mm(tmp, lora_b);

        auto res = out + x;
        return {res};
    }

};

#endif // MODELING_TESTLORA_SIDEINFERENCE_HPP