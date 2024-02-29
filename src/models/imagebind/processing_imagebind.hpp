//
// Created by ey on 24-2-29.
//

#ifndef PROCESSING_IMAGEBIND_HPP
#define PROCESSING_IMAGEBIND_HPP
#include <utility>

#include "processor/ClipPreProcess.hpp"
#include "tokenizers/BPE/Bpe.hpp"

using namespace mllm;

class ImagebindProcessor final {
    BPETokenizer *tokenizer;
    ClipPreProcessor *clip_processor;
    static Tensor tokens2Input(vector<vector<token_id_t>> tokens, int max_pos, string name = "input", BackendType type = MLLM_CPU) {
        const auto bsize = static_cast<int>(tokens.size());
        Tensor tensor1(bsize, 1, max_pos, 1, Module::backends[type], true);
        tensor1.setName(name);
        tensor1.status() = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int b = 0; b < bsize; ++b) {
            for (int idx = 0; idx < tokens[b].size(); ++idx) {
                tensor1.setDataAt<float>(b, 0, idx, 0, tokens[b][idx]);
            }
        }
        return tensor1;
    }
    static Tensor img2Tensor(vector<vector<vector<vector<float>>>> imgs, string name = "input", BackendType type = MLLM_CPU) {
        int channel = imgs[0].size();
        int height = imgs[0][0].size();
        int width = imgs[0][0][0].size();
        Tensor tensor1(Module::backends[type]);
        tensor1.reshape(imgs.size(), channel, 2, height, width);
        tensor1.setDtype(MLLM_TYPE_F32);
        tensor1.alloc();
        tensor1.setName(std::move(name));
        tensor1.status() = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int bi = 0; bi < imgs.size(); ++bi) {
            for (int t = 0; t < 2; ++t) {
                for (int h = 0; h < height; ++h) {
                    for (int c = 0; c < channel; ++c) {
                        for (int w = 0; w < width; ++w) {
                            tensor1.setDataAt<float>(bi, c, t, h, w, imgs[bi][c][h][w]);
                        }
                    }
                }
            }
        }
        return tensor1;
    }
    static Tensor audio2Tensor(vector<vector<vector<vector<float>>>> audio, string name = "input", BackendType type = MLLM_CPU) {
        vector<vector<vector<float>>> audio_new;
        for (auto auv : audio) {
            for (auto au : auv) {
                audio_new.push_back(au);
            }
        }
        int batch = audio_new.size();
        int channel = 1;
        int height = audio_new[0].size();
        int width = audio_new[0][0].size();

        Tensor tensor1(batch, height, channel, width, Module::backends[type], true);
        tensor1.setName(std::move(name));
        tensor1.status() = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);

        for (int bi = 0; bi < audio_new.size(); ++bi) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    tensor1.setDataAt<float>(bi, h, 0, w, audio_new[bi][h][w]);
                }
            }
        }
        return tensor1;
    }

public:
    explicit ImagebindProcessor(const string &vocab_path, const string &merges_path) {
        tokenizer = new BPETokenizer(vocab_path);
        std::unordered_map<string, unsigned> merge_rank;
        auto merge_file = std::ifstream(merges_path);
        std::string line;
        unsigned rank = 0;
        while (std::getline(merge_file, line)) {
            if (line.empty()) {
                continue;
            }
            if (line[0] == '#') {
                continue;
            }
            merge_rank[line] = rank;
            rank++;
        }
        tokenizer->setMergeRank(merge_rank);
        tokenizer->setSpecialToken("<|startoftext|>", "<|endoftext|>");
        clip_processor = new ClipPreProcessor(tokenizer);
    }

    std::tuple<std::array<Tensor, 3>, int> process(vector<string> in_strs, int max_pos, vector<string> img_path, int hw, vector<string> wav_path,
                                                   string text_name = "input_text", string img_name = "input_vision", string wav_name = "input_audio",
                                                   BackendType type = MLLM_CPU) {
        auto tokens_ids = vector<vector<token_id_t>>();
        for (auto in_str : in_strs) {
            vector<mllm::token_id_t> tokens_id = {};
            tokenizer->tokenize(in_str, tokens_id, true, true, "</w>");
            tokens_ids.push_back(tokens_id);
        }
        int input_text_lens = tokens_ids[0].size() - 1;

        clip_processor->PreProcessImages(img_path, hw, hw);
        auto images = clip_processor->pixel_values_;

        auto audios = PreProcessor::ProcessAudio(std::move(wav_path));

        return {{tokens2Input(tokens_ids, max_pos, std::move(text_name)),
                 img2Tensor(images, std::move(img_name)),
                 audio2Tensor(audios, std::move(wav_name))},
                input_text_lens};
    }
};

#endif // PROCESSING_IMAGEBIND_HPP
