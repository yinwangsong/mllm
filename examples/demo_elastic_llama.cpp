//
// Created by Rongjie Yi on 2024/1/26 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/llama/modeling_elastic_llama.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "processor/PostProcess.hpp"


using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llama-2-7b-chat-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 1600);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = LLaMATokenizer(vocab_path);

    LLaMAConfig config(tokens_limit, "7B", LLAMAROPE);
    auto model = ElasticLLaMAModel(config);
    model.load(model_path);

    // vector<string> in_strs = {
    //     " Hello, who are you?",
    //     " What can you do?",
    //     "Please introduce Beijing University of Posts and Telecommunications."};


    int input_lens[10] = {1, 25, 50, 75, 100, 125, 150, 175, 200, 225};
    // int input_lens[10] = {1, 50, 100, 150, 200, 250, 300, 350, 400, 450};
    // int input_lens[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    // int input_lens[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float model_size[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
    for (int i = 0; i < 10; ++i) {
        // auto in_str = in_strs[i];
        // auto input_tensor = tokenizer.tokenize(in_str, i);
        // std::cout << "[Q] " << in_str << std::endl;
        // std::cout << "[A] " << std::flush;

        // int seq_len = input_lens[i];
        int seq_len = 100;
        auto input_tensor = Tensor(1, 1, seq_len, 1,Module::backends[MLLM_CPU], true);
        input_tensor.setName("input_ids");
        input_tensor.setTtype(INPUT_TENSOR);
        for (int idx = 0; idx < seq_len; ++idx) {
            input_tensor.setDataAt<float>(0, 0, idx, 0, 1);
        }

        for (int step = 0; step < 10; step++) {
            // vecor<vector<int>> activate_dims = {{32*8,256}}; 
            // 32*8 is attn_head*attn_hidden_dim(e.g. llama:32*128); 256 is ffn_hidden_dim(e.g. llama:11008) 
            // vector<vector<int>> activate_dims = {
            //                             {-1,-1},  //0
            //                             {-1,-1}, //1
            //                             {-1,-1}, //2
            //                             {-1,-1}, //3
            //                             {-1,-1}, //4
            //                             {-1,-1}, //5
            //                             {-1,-1}, //6
            //                             {-1,-1}, //7
            //                             {-1,-1}, //8
            //                             {-1,-1}, //9
            //                             {-1,-1}, //10
            //                             {-1,-1}, //11
            //                             {-1,-1}, //12
            //                             {-1,-1}, //13
            //                             {-1,-1}, //14
            //                             {-1,-1}, //15
            //                             {-1,-1}, //16
            //                             {-1,-1}, //17
            //                             {-1,-1}, //18
            //                             {-1,-1}, //19
            //                             {-1,-1}, //20
            //                             {-1,-1}, //21
            //                             {-1,-1}, //22
            //                             {-1,-1}, //23
            //                             {-1,-1}, //24
            //                             {-1,-1}, //25
            //                             {-1,-1}, //26
            //                             {-1,-1}, //27
            //                             {-1,-1}, //28
            //                             {-1,-1}, //29
            //                             {-1,-1}, //30
            //                             {-1,-1} //31
            // };

            float model_ratio = model_size[i];

            vector<vector<int>> activate_dims = {
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
                                        {int(32*128*model_ratio/256)*256, int(11008*model_ratio/256)*256},  //0
            };
            auto start = std::chrono::high_resolution_clock::now();

            auto result = model({input_tensor}, activate_dims);

            auto end = std::chrono::high_resolution_clock::now();

            // 计算时间差
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            // 输出执行时间
            std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

            auto outputs = tokenizer.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            if (out_token == 2) {
                break;
            }
            std::cout << out_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
    }

    return 0;
}