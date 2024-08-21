//
// Created by Rongjie Yi on 2024/1/26 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/simulated_mobilebert/modeling_llama.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"
#include <fstream>
#include <string>
#include <thread>
#include <chrono>

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llama-2-7b-chat-q4_0.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 10240);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);

    cmdParser.add<int>("prompt_len", 'p', "# of input tokens", false, 1);
    cmdParser.add<float>("model_size", 's', "model size", false, 0.1);

    cmdParser.add<int>("model_type", 'b', "mobilebert or roberta", false, 1);

    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    int seq_len = cmdParser.get<int>("prompt_len");
    int model_type = cmdParser.get<int>("model_type");


    auto tokenizer = LLaMATokenizer(vocab_path);

    string bi = "mobilebert";
    if(model_type == 1){
        bi = "mobilebert";
    }
    else{
        bi = "roberta";
    }

    LLaMAConfig config(tokens_limit, bi, LLAMAROPE);
    auto model = LLaMAModel(config);


    // model.load(model_path);

    auto input_tensor = Tensor(1, 1, seq_len, 1, Module::backends[MLLM_CPU], true);
    input_tensor.setName("input_ids");
    input_tensor.setTtype(INPUT_TENSOR);
    for (int idx = 0; idx < seq_len; ++idx) {
        input_tensor.setDataAt<float>(0, 0, idx, 0, 1);
    }

    auto tmp = model({input_tensor});

    for (int step = 0; step < 1; step++) {
        // warm up

        auto start = std::chrono::high_resolution_clock::now();

        auto result = model({input_tensor});

        auto end = std::chrono::high_resolution_clock::now();

        // 计算时间差
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "input_len: " << seq_len << " Execution time: " << duration.count() << " ms" << std::endl;

        auto outputs = tokenizer.detokenize(result[0]);
        auto out_string = outputs.first;
        auto out_token = outputs.second;
        if (out_token == 2) {
            break;
        }
        std::cout << out_string << std::flush;
        // chatPostProcessing(out_token, input_tensor, {});
    }
    printf("\n");
    return 0;
}
