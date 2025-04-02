model_path="/home/slurm/tuenv2/open_i1_project/sft/360-LLaMA-Factory/exp_r1/sft-1.5b-qwen-math-base-full-packing-math-decontaminated-v1/checkpoint-2000"
model_name="ii-sft-decontaminated-v1-2000"
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