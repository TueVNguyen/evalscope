model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
num_gpus=1
host="0.0.0.0"
port=1234
cuda_device=0,1,2,3,4,5,6,7

CUDA_VISIBLE_DEVICES=$cuda_device python -m sglang.launch_server \
        --model $model_path \
        --trust-remote-code \
        --served-model-name $model_name \
        --tensor-parallel-size $num_gpus \
        --port $port \
        --host $host \
        --dp 8