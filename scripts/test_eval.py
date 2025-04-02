from evalscope import TaskConfig, run_task

import os

import sys


from huggingface_hub import login

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

parser.add_argument("--work_dir", type=str, default="outputs")

parser.add_argument("--api_url", type=str, default="http://0.0.0.0:1234/v1")
parser.add_argument("--max_tokens", type=int, default=32768)
args = parser.parse_args()


# 64 samples
# "aime25_full",
task_config_64 = TaskConfig(
    api_url=args.api_url, # Inference service address
    model=args.model_name, # Model name (must match the deployed model name)
    eval_type='service', # Evaluation type; SERVICE indicates evaluating the inference service
    datasets=[ "aime24", "amc23", "aime25_full"], # Dataset name
    # dataset_args={'aime25_full': {'few_shot_num': 0}}, # Dataset parameters
    eval_batch_size=16, # Number of concurrent requests
    generation_config={
        'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
        'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
        'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
        'top_k': 40, # .Top-k sampling (recommended value from Qwen)
        'n': 64,
        "seed": 42,
    #  of responses generated for each request
    },
    work_dir=args.work_dir,
    judge_worker_num=1
)

task_config_8 = TaskConfig(

    api_url=args.api_url, # Inference service address
    model=args.model_name, # Model name (must match the deployed model name)
    eval_type='service', # Evaluation type; SERVICE indicates evaluating the inference service
    datasets=['minerva_math', 'math_gakao_en' ], # Dataset name
    eval_batch_size=32, # Number of concurrent requests
    generation_config={
        'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
        'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
        'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
        'top_k': 40, # .Top-k sampling (recommended value from Qwen)
        'n': 4, # Number of responses generated for each request,
        "seed": 42,
    },
    work_dir=args.work_dir,
    judge_worker_num=1
)

task_config_4 =TaskConfig(
    api_url=args.api_url, # Inference service address
    model=args.model_name, # Model name (must match the deployed model name)
    eval_type='service', # Evaluation type; SERVICE indicates evaluating the inference service
    datasets=[ 'math_500', 'olympiad_bench', 'ii_math_entrance_exam'], # Dataset name
    eval_batch_size=128, # Number of concurrent requests
    generation_config={

        'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
        'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
        'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
        'top_k': 40, # .Top-k sampling (recommended value from Qwen)
        'n': 4, # Number of responses generated for each request
        "seed": 42,

    },
    work_dir=args.work_dir,
    judge_worker_num=1
)
task_config_1 =TaskConfig(
    api_url=args.api_url, # Inference service address
    model=args.model_name, # Model name (must match the deployed model name)
    eval_type='service', # Evaluation type; SERVICE indicates evaluating the inference service
    datasets=[ 'ifeval'], # Dataset name
    eval_batch_size=512, # Number of concurrent requests
    generation_config={

        'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
        'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
        'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
        'top_k': 40, # .Top-k sampling (recommended value from Qwen)
        'n': 1, # Number of responses generated for each request
        "seed": 42,

    },
    work_dir=args.work_dir,
    judge_worker_num=1
)
task_config_0 =TaskConfig(
    api_url=args.api_url, # Inference service address
    model=args.model_name, # Model name (must match the deployed model name)
    eval_type='service', # Evaluation type; SERVICE indicates evaluating the inference service
    datasets=[ 'live_code_bench_reasoning'], # Dataset name
    eval_batch_size=256, # Number of concurrent requests
    generation_config={

        'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
        'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
        'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
        'top_k': 40, # .Top-k sampling (recommended value from Qwen)
        'n': 8, # Number of responses generated for each request
        "seed": 42,

    },
    work_dir=args.work_dir,
    judge_worker_num=1
)
# ru
run_task([
   task_config_64,
   task_config_8,
  task_config_4,
#   task_config_1
])
# 