#include <csignal>
#include "Executor.hpp"
namespace mllm {
void Executor::init() {
}

// #define DEBUG
// #define DYNAMIC
bool paramloaded = false;
bool freeGraph = false;
void Executor::execute(Net *net, shared_ptr<Tensor> input_tensor) {
    auto input_size = input_tensor->shape();
    bool init = false;
    bool reshape = false;
    // TODO: when reshape begin
    checkReshape(init, reshape, input_size);
    // set Input tensor

    uint64_t t_start;
    uint64_t t_end;
    uint64_t time_start = mllm_time_us();
    uint64_t time_end;

    input_tensor->setName(net->inputName());
    net->tensors()[net->inputName()] = input_tensor;
    net->subGraph()["G0"]->reflashInput(net->tensors(), net->inputName());
    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net->subGraph()[name];
        if (init || reshape) {
#ifdef DEBUG
            std::cout << "[" << name << "]==== reshape";
            t_start = mllm_time_us();
#endif
            g->reshape();
#ifdef DEBUG
            t_end = mllm_time_us();
            std::cout << " ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
        }
        // load params
        if (!paramloaded) {
#ifdef DEBUG
            std::cout << "[" << name << "]==== load";
            t_start = mllm_time_us();
#endif
            g->setUpOps(*data_loader_);
#ifdef DEBUG
            t_end = mllm_time_us();
            std::cout << "    ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
        }
#ifndef DYNAMIC
    }
    paramloaded = true;
    time_end = mllm_time_us();
    if (load_time_ == 0) {
        load_time_ = (time_end - time_start) / 1000.0F;
    }
#ifdef DEBUG
    std::cout << "reshape&load ====  " << (time_end - time_start) / 1000.0F << " ms" << std::endl;
#endif
    time_start = mllm_time_us();

    for (int i = 0; i < (int)net->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net->subGraph()[name];
#endif
        // if (init || reshape)
//        {
#ifdef DEBUG
        std::cout << "[" << name << "]==== setup";
        t_start = mllm_time_us();
#endif
        g->reshape();
        g->setUpTensors();
#ifdef DEBUG
        t_end = mllm_time_us();
        std::cout << " ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
//        }
// exe
#ifdef DEBUG
        std::cout << "[" << name << "]==== execute";
        t_start = mllm_time_us();
#endif
        result_ = g->forward();
#ifdef DEBUG
        t_end = mllm_time_us();
        std::cout << " ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
        // free
        if (freeGraph) {
#ifdef DEBUG
            std::cout << "[" << name << "]==== free";
            t_start = mllm_time_us();
#endif
#ifdef DYNAMIC
            g->freeOps();
            paramloaded = false;
#endif
            if (i < (int)net->subGraph().size() - 1) {
                g->freeTensors();
            }
            net->freeTensors(i);
#ifdef DEBUG
            t_end = mllm_time_us();
            std::cout << "    ====  " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif
        }
        // std::cout <<"["<< name << "]==== end      === "<< result_[0]->name() << "'s shape:  [" << result_[0]->shape(0) << "," << result_[0]->shape(1) << "," << result_[0]->shape(2) << "," << result_[0]->shape(3) << "]" << std::endl;
    }
    time_end = mllm_time_us();
    if (input_size[2] == 1) {
        auto token_run_time = (time_end - time_start) / 1000.0F;
        run_time_ += token_run_time;
        run_times_ += 1;
    }
#ifdef DEBUG
    std::cout << "exec ====  " << (time_end - time_start) / 1000.0F << " ms" << std::endl;
#endif
}

} // namespace mllm
