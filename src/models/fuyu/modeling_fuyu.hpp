//
// Created by Rongjie Yi on 2024/2/14 0004.
//

#ifndef MODELING_FUYU_HPP
#define MODELING_FUYU_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_fuyu.hpp"

#include <models/transformer/modeling_transformer.hpp>

using namespace mllm;

class PersimmonBlock final : public Module {
    MultiHeadAttention attention;
    FeedForward mlp;
    Layer norm1;
    Layer norm2;

public:
    PersimmonBlock() = default;
    PersimmonBlock(int hidden_dim, int head_size, int ffn_hidden, int cache_limit, const FuyuNameConfig &names, const string &base_name) {
        attention = MultiHeadAttention(hidden_dim, head_size, hidden_dim / head_size, true, true, false,
                                       PERSIMMONROPE, cache_limit, true, true, names, base_name + names._attn_base_name);
        mlp = FeedForward(hidden_dim, ffn_hidden, "ReLU2", true,
                          names, base_name + names._ffn_base_name);
        norm1 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._attn_norm_name);
        norm2 = LayerNorm(hidden_dim, true, 1e-6, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override  {
        auto x = norm1(inputs[0]);
        x = attention({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }
};

class Persimmon final : public Module {
    vector<PersimmonBlock> blocks;
    Layer norm;
    Layer lm_head;

public:
    Persimmon() = default;
    Persimmon(int hidden_dim, int head_size, int ffn_hidden, int cache_limit, int block_num, int vocab_size, const FuyuNameConfig &names) {
        blocks = List<PersimmonBlock>(block_num, hidden_dim, head_size, ffn_hidden, cache_limit, names, names.blk_name);
        norm = LayerNorm(hidden_dim, true, 1e-6, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override  {
        auto x = inputs[0];
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }
};

class FuyuGather final : public Layer {
public:
    FuyuGather() = default;
    explicit FuyuGather(std::string name) {
        init(std::move(name), OpType::GATHER);
    }
    Tensor &operator()(Tensor &input_ids, Tensor &image_patches, Tensor &image_patches_indices) {
        return _3I1O_OP(input_ids, image_patches, image_patches_indices);
    }
};

class FuyuModel final : public Module {
    Layer embed_tokens;
    Layer vision_embed_tokens;
    FuyuGather fuyu_gather;
    Persimmon persimmon;

public:
    explicit FuyuModel(const FuyuConfig &config) :
        FuyuModel(config.vocab_size, config.hidden_dim, config.head_size, config.ffn_hidden, config.block_num,
                  config.cache_limit, config.patch_size, config.chl_size,
                  config.name_config) {
    }
    FuyuModel(int vocab_size, int hidden_dim, int head_size, int ffn_hidden, int block_num,
              int cache_limit, int patch_size, int chl_size,
              const FuyuNameConfig &names) {
        embed_tokens = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        vision_embed_tokens = Linear(patch_size * patch_size * chl_size, hidden_dim, true, names.vision_embed_tokens_name);
        fuyu_gather = FuyuGather("gather");
        persimmon = Persimmon(hidden_dim, head_size, ffn_hidden, cache_limit, block_num, vocab_size, names);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override  {
        auto input_ids = embed_tokens(inputs[0]);
        if (inputs[1].batch() > 0) {
            auto image_patches = vision_embed_tokens(inputs[1]);
            input_ids = fuyu_gather(input_ids, image_patches, inputs[2]);
        }
        return persimmon({input_ids});
    }
};

#endif // MODELING_FUYU_HPP