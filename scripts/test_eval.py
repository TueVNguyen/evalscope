from evalscope import TaskConfig, run_task
import os

task_config = TaskConfig(
    api_url='http://0.0.0.0:2596/v1',  # Inference service address
    model='gemini_flash_2.0',  # Model name (must match the deployed model name)
    eval_type='service',  # Evaluation type; SERVICE indicates evaluating the inference service
    datasets=["aime24",'aime25_full', 'olympiad_bench', 'minerva_math', 'math_gakao_en', 'ii_math_entrance_exam', 'amc23'],  # Dataset name
    dataset_args={'aime25_full': {'few_shot_num': 0}},  # Dataset parameters
    eval_batch_size=64,  # Number of concurrent requests
    generation_config={
        'max_tokens': 16384,  # Maximum number of tokens to generate; recommended to set a high value to avoid truncation
        'temperature': 0.6,  # Sampling temperature (recommended value from Qwen)
        'top_p': 0.95,  # Top-p sampling (recommended value from Qwen)
        # 'top_k': 40,  # Top-k sampling (recommended value from Qwen)
        'n': 2,  # Number of responses generated for each request
    },
    # dataset_dir=os.path.join(os.path.expanduser('~'), '.cache/huggingface/datasets'),
    # dataset_hub="huggingface",
)
run_task(task_config)