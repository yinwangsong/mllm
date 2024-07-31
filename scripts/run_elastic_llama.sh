#!/bin/bash

adb.exe shell mkdir /data/local/tmp/mllm
adb.exe shell mkdir /data/local/tmp/mllm/bin
# adb shell mkdir /data/local/tmp/mllm/models
adb.exe shell mkdir /data/local/tmp/mllm/vocab
adb.exe push ../vocab/llama_vocab.mllm /data/local/tmp/mllm/vocab/
#adb push ../bin-arm/main_llama /data/local/tmp/mllm/bin/
adb.exe push ../bin-arm/demo_elastic_llama /data/local/tmp/mllm/bin/

# adb push ../models/llama-2-7b-chat-q4_k.mllm /data/local/tmp/mllm/models/
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
#adb shell "cd /data/local/tmp/mllm/bin && ./main_llama"
# adb.exe shell "cd /data/local/tmp/mllm/bin && chmod 777 ./demo_elastic_llama && ./demo_elastic_llama --thread 1"

# adb.exe pull /data/local/tmp/mllm/bin/prefill_llama7b.txt .
# adb.exe pull /data/local/tmp/mllm/bin/decode_llama7b.txt .

# 创建数组
prompt_len=(1 25 50 75 100 125 150 175 200 225)
model_size=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

adb.exe shell "cd /data/local/tmp/mllm/bin && > prefill_llama7b.txt && > decode_llama7b.txt"

for ((i=0; i<${#prompt_len[@]}; i++)); do
for ((j=0; j<${#model_size[@]}; j++)); do
    adb.exe shell "cd /data/local/tmp/mllm/bin && chmod 777 ./demo_elastic_llama && ./demo_elastic_llama --prompt_len ${prompt_len[$j]} --model_size ${model_size[$i]}"
done
done
adb.exe pull /data/local/tmp/mllm/bin/prefill_llama7b.txt .
adb.exe pull /data/local/tmp/mllm/bin/decode_llama7b.txt .