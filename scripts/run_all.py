from evalscope import TaskConfig, run_task
import os
import sys
import json
import requests
from huggingface_hub import login
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

parser.add_argument("--work_dir", type=str, default="outputs")

parser.add_argument("--api_url", type=str, default="http://0.0.0.0:1234/v1")
parser.add_argument("--max_tokens", type=int, default=32768)
parser.add_argument("--cache_dir", type=str, default=None, help="If provided, the cache results model response before will be used and we only run evaluate only")
parser.add_argument("--system_prompt", type=str, default=None)
parser.add_argument("--eval_batch_size", type=int, default=512)
args = parser.parse_args()
if __name__ == "__main__":
    # Math level
    # First, we dump info model from sglang 
    api_url = args.api_url
    url = api_url.rstrip('/').rsplit('/chat/completions', 1)[0].rsplit('/v1', 1)[0] + '/get_model_info'
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(args.work_dir, exist_ok=True)
        with open(args.work_dir + "/model_info.json", "w") as f:
            json.dump(response.json(), f, indent=4)
            args.model_name = response.json()["model_path"]
    system_prompt = args.system_prompt
    task_config = TaskConfig(
        api_url=args.api_url, # Inference service address
        model=args.model_name, # Model name (must match the deployed model name)
        eval_type='service', # Evaluation type; SERVICE indicates evaluating the inference service
        datasets=[ "aime24", "amc23", "aime25_full", "math_500", "minerva_math", "math_gakao_en", "olympiad_bench", "ii_math_entrance_exam",
                    "ifeval", "gpqa", "mmlu", "live_code_bench"],
        dataset_args={
            "gpqa": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 4,
                    "seed": 42,
                }
            },
            "aime24": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 64,
                    "seed": 42,
                }
            },
            "amc23": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 64,
                    "seed": 42, 
                }
            },
            "aime25_full": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 64,
                    "seed": 42,
                }
            },
            "math_500": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 4,
                    "seed": 42,
                }
            },  
            "minerva_math": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 4,
                    "seed": 42,
                }
            },
            "math_gakao_en": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 4,
                    "seed": 42,
                }
            },
            "olympiad_bench": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 4,
                    "seed": 42,
                }
            },
            "ii_math_entrance_exam": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, #   Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 4,
                    "seed": 42,
                }
            },
            "ifeval": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 1,
                    "seed": 42,
                }
            },
            "live_code_bench": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 8,
                    "seed": 42,
                },

                "judge_worker_num": 32 # for live_code_bench, we need to set a high value to avoid timeout code
            },
            "mmlu": {
                "generation_config": {
                    'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
                    'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
                    'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
                    'n': 1,
                    "seed": 42,
                }
            }
        },
        eval_batch_size=args.eval_batch_size, # Number of concurrent requests
        generation_config={
            'max_tokens': args.max_tokens, # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
            'temperature': 0.6, # Sampling temperature (recommended value from Qwen)
            'top_p': 0.95, # Top-p sampling (recommended value from Qwen)
            'n': 64,
            "seed": 42,
        },
        work_dir=args.work_dir,
        judge_worker_num=1,
        use_cache=None if args.cache_dir is None else args.cache_dir # it is equally to args.cache_dir...
    )
    run_task(task_config)