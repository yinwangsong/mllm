#!/bin/bash

adb.exe shell mkdir /data/local/tmp/mllm
adb.exe shell mkdir /data/local/tmp/mllm/bin
# adb shell mkdir /data/local/tmp/mllm/models
adb.exe shell mkdir /data/local/tmp/mllm/vocab
adb.exe push ../vocab/llama_vocab.mllm /data/local/tmp/mllm/vocab/
#adb push ../bin-arm/main_llama /data/local/tmp/mllm/bin/
adb.exe push ../bin-arm/demo_test_llama_reorder_lora_overhead /data/local/tmp/mllm/bin/

# adb push ../models/llama-2-7b-chat-q4_k.mllm /data/local/tmp/mllm/models/
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
#adb shell "cd /data/local/tmp/mllm/bin && ./main_llama"
adb.exe shell "cd /data/local/tmp/mllm/bin && chmod 777 ./demo_test_llama_reorder_lora_overhead && ./demo_test_llama_reorder_lora_overhead"