//
// Created by Rongjie Yi on 2024/1/26 0026.
//

#include <iostream>
#include "cmdline.h"
#include "models/llama/modeling_elastic_llama.hpp"
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
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llama-2-7b-chat-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 10240);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);

    cmdParser.add<int>("prompt_len", 'p', "# of input tokens", false, 1);
    cmdParser.add<float>("model_size", 's', "model size", false, 0.1);

    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    int seq_len = cmdParser.get<int>("prompt_len");
    float model_ratio = cmdParser.get<float>("model_size");

    auto tokenizer = LLaMATokenizer(vocab_path);

    LLaMAConfig config(tokens_limit, "7B", LLAMAROPE);
    auto model = ElasticLLaMAModel(config);
    model.load(model_path);


    std::string prefil_filename = "prefill_llama7b.txt";
    // 使用ofstream对象来写入文件
    std::ofstream prefill_file(prefil_filename, std::ios_base::app);

    // 检查文件是否成功打开
    if (!prefill_file.is_open()) {
        std::cerr << "Failed to open file: " << prefil_filename << std::endl;
        return 1; // 返回错误代码
    }

    std::string decode_filename = "decode_llama7b.txt";
    // 使用ofstream对象来写入文件
    std::ofstream decode_file(decode_filename, std::ios_base::app);

    // 检查文件是否成功打开
    if (!decode_file.is_open()) {
        std::cerr << "Failed to open file: " << decode_filename << std::endl;
        return 1; // 返回错误代码
    }

    auto input_tensor = Tensor(1, 1, seq_len, 1,Module::backends[MLLM_CPU], true);
    input_tensor.setName("input_ids");
    input_tensor.setTtype(INPUT_TENSOR);
    for (int idx = 0; idx < seq_len; ++idx) {
        input_tensor.setDataAt<float>(0, 0, idx, 0, 1);
    }

    for (int step = 0; step < 3; step++) {
        // vecor<vector<int>> activate_dims = {{32*8,256}}; 
        // 32*8 is attn_head*attn_hidden_dim(e.g. llama:32*128); 256 is ffn_hidden_dim(e.g. llama:11008) 

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
        
        // warm up
        auto tmp = model({input_tensor}, activate_dims);

        auto start = std::chrono::high_resolution_clock::now();

        auto result = model({input_tensor}, activate_dims);

        auto end = std::chrono::high_resolution_clock::now();

        // 计算时间差
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "input_len: " << seq_len << " model_size " << model_ratio << " Execution time: " << duration.count() << " ms" << std::endl;


        if(step==0){
            // 写入一行文本
            prefill_file << seq_len << " " << model_ratio << " " << duration.count() << std::endl;
        }
        if(step==2){
            // 写入一行文本
            decode_file << seq_len << " " << model_ratio << " " << duration.count() << std::endl;
        }


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

    // 关闭文件
    prefill_file.close();
    decode_file.close();
    return 0;
}