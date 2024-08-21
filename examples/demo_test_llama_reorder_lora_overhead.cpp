// //
// // Created by Wangsong Yin on 2024/8/3 0531.
// //

/**
 * since currently there is no support for fp16 in mllm, and fp32 will be OOM, so we simply perform an one-operator test here
 */


#include <iostream>
#include "cmdline.h"
#include "models/testlora/modeling_testlora.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "models/testlora/modeling_testlora_sideinference.hpp"
#include "processor/PostProcess.hpp"
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <algorithm>
#include <random>


using namespace mllm;

int main(int argc, char **argv) {

    Module::initBackend(MLLM_CPU);

    cmdline::parser cmdParser;
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);

    cmdParser.add<int>("rank", 'r', "num of rank", false, 8);

    cmdParser.add<int>("indim", 'i', "input dims", false, 4096);
    cmdParser.add<int>("seqlen", 'l', "seq len", false, 512);
    cmdParser.add<int>("outdim", 'o', "output dims", false, 4096);
    cmdParser.parse_check(argc, argv);

    CPUBackend::cpu_threads = cmdParser.get<int>("thread");


    int in_dim = cmdParser.get<int>("indim");
    int out_dim = cmdParser.get<int>("outdim");
    int seqlen = cmdParser.get<int>("seqlen");
    
    // generate a random 4096 index array for reordering.
    std::vector<int> numbers;
    for (int i = 0; i < in_dim; ++i) {
        numbers.push_back(i);
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(numbers.begin(), numbers.end(), g);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // create a 4096*4096 matrix
    // auto tensor_for_reorder = Tensor(1, 1, in_dim, out_dim, Module::backends[MLLM_CPU], true);
    // for (int idx1 = 0; idx1 < in_dim; ++idx1) {
    //     for (int idx2 = 0; idx2 < out_dim; ++idx2) {
    //         tensor_for_reorder.setDataAt<float>(0, 0, idx1, idx2, dis(g));
    //     }
    // }
    auto tensor_for_reorder = Tensor(1, 1, in_dim, out_dim, Module::backends[MLLM_CPU], true);
    for (int idx1 = 0; idx1 < in_dim; ++idx1) {
        for (int idx2 = 0; idx2 < out_dim; ++idx2) {
            tensor_for_reorder.setDataAt<float>(0, 0, idx1, idx2, dis(g));
        }
    }


    auto start = std::chrono::high_resolution_clock::now();

    // since mllm currently does not support in-pace swap of tensor data elements, we apply a buffer to test reorder.

    auto reoder_buffer = Tensor(1, 1, in_dim, out_dim, Module::backends[MLLM_CPU], true);
    reoder_buffer.copyFrom(tensor_for_reorder);

    // perform reorder
    for (int idx1 = 0; idx1 < in_dim; ++idx1) {
        for (int idx2 = 0; idx2 < out_dim; ++idx2) {
            tensor_for_reorder.setDataAt<float>(0, 0, idx1, idx2, reoder_buffer.dataAt<float>(0, 0, numbers[idx1], idx2));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << " shuffle time: " << duration.count() << " ms" << std::endl;

    // int seq_len = 1;
    // auto input_tensor = Tensor(1, 1, seq_len, 1, Module::backends[MLLM_CPU], true);
    // input_tensor.setName("input_ids");
    // input_tensor.setTtype(INPUT_TENSOR);
    // for (int idx = 0; idx < seq_len; ++idx) {
    //     input_tensor.setDataAt<float>(0, 0, idx, 0, 1);
    // }

    // create a 4096*4096 matrix
    auto tensor_for_lora = Tensor(1, 1, seqlen, out_dim, Module::backends[MLLM_CPU], true);
    tensor_for_lora.setTtype(INPUT_TENSOR);
    for (int idx1 = 0; idx1 < seqlen; ++idx1) {
        for (int idx2 = 0; idx2 < out_dim; ++idx2) {
            tensor_for_lora.setDataAt<float>(0, 0, idx1, idx2, dis(g));
        }
    }

    int r = cmdParser.get<int>("rank");

    // creare loras
    auto lora_a = Tensor(1, 1, in_dim,r , Module::backends[MLLM_CPU], true);
    lora_a.setTtype(INPUT_TENSOR);
    for (int idx1 = 0; idx1 < in_dim; ++idx1) {
        for (int idx2 = 0; idx2 < r; ++idx2) {
            lora_a.setDataAt<float>(0, 0, idx1, idx2, dis(g));
        }
    }

    auto lora_b = Tensor(1, 1, r, out_dim, Module::backends[MLLM_CPU], true);
    lora_b.setTtype(INPUT_TENSOR);
    for (int idx1 = 0; idx1 < r; ++idx1) {
        for (int idx2 = 0; idx2 < out_dim; ++idx2) {
            lora_b.setDataAt<float>(0, 0, idx1, idx2, dis(g));
        }
    }

    auto model = testlora();

    auto start2 = std::chrono::high_resolution_clock::now();

    model({tensor_for_lora, lora_a, lora_b});

    auto end2 = std::chrono::high_resolution_clock::now();


    // 计算时间差
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout<< "lora time:" << duration2.count() <<" ms\n";


    // int seq_len = 1;
    // auto input_tensor = Tensor(1, 1, seq_len, 1, Module::backends[MLLM_CPU], true);
    // input_tensor.setName("input_ids");
    // input_tensor.setTtype(INPUT_TENSOR);
    // for (int idx = 0; idx < seq_len; ++idx) {
    //     input_tensor.setDataAt<float>(0, 0, idx, 0, 1);
    // }

    // auto x = Tensor(1, 1, 1, in_dim, Module::backends[MLLM_CPU], true);
    // x.setTtype(INPUT_TENSOR);
    // for (int idx1 = 0; idx1 < 1; ++idx1) {
    //     for (int idx2 = 0; idx2 < in_dim; ++idx2) {
    //         x.setDataAt<float>(0, 0, idx1, idx2, dis(g));
    //     }
    // }


    // auto model2 = testlora_sideinference();

    // auto start3 = std::chrono::high_resolution_clock::now();

    // model2({x, lora_a, lora_b});

    // auto end3 = std::chrono::high_resolution_clock::now();


    // // 计算时间差
    // auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    // std::cout<< "detached inference time:" << duration3.count();

    // tensor_for_reorder.printData<float>();
    return 0;
}