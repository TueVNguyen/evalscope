from evalscope.third_party.thinkbench import run_task
from evalscope.third_party.thinkbench.eval import combine_results

judge_config = dict(  # Evaluation service configuration
    api_key='EMPTY',
    base_url='http://0.0.0.0:4000/v1',
    model_name='gpt-4o',
    
)

model_config = dict(
    report_path='/home/slurm/tuenv2/open_i1_project/evaluation/evalscope/all/2',  # Path to the model reasoning results from the previous step
    model_name='DeepSeek-R1-Distill-Qwen-1.5B',  # Model name
    tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',  # Path to the model tokenizer for token count calculation
    dataset_name='math_500',  # Dataset name from the previous step
    subsets=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],  # Subsets from the previous step
    split_strategies='separator',  # Strategy for splitting reasoning steps; options are separator, keywords, llm
    judge_config=judge_config,
)
model_config2 = dict(
    report_path='/home/slurm/tuenv2/open_i1_project/evaluation/evalscope/final_benchmark_results/aduy_21_03_2025_global_step_400/20250321_120427/1',  # Path to the model reasoning results from the previous step
    model_name='aduy',  # Model name
    tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',  # Path to the model tokenizer for token count calculation
    dataset_name='math_500',  # Dataset name from the previous step
    subsets=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],  # Subsets from the previous step
    split_strategies='separator',  # Strategy for splitting reasoning steps; options are separator, keywords, llm
    judge_config=judge_config,
)
model_config3 = dict(
    report_path='/home/slurm/tuenv2/open_i1_project/evaluation/evalscope/all/5',  # Path to the model reasoning results from the previous step
    model_name='ii-rl-math-code-v1',  # Model name
    tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',  # Path to the model tokenizer for token count calculation
    dataset_name='math_500',  # Dataset name from the previous step
    subsets=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],  # Subsets from the previous step
    split_strategies='separator',  # Strategy for splitting reasoning steps; options are separator, keywords, llm
    judge_config=judge_config,
)
model_config4 = dict(
    report_path='/home/slurm/tuenv2/open_i1_project/evaluation/evalscope/final_benchmark_results/rl/Aduy8K_32max_tokens/20250325_074849/0',  # Path to the model reasoning results from the previous step
    model_name='ADuy_8k',  # Model name
    tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',  # Path to the model tokenizer for token count calculation
    dataset_name='math_500',  # Dataset name from the previous step
    subsets=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],  # Subsets from the previous step
    split_strategies='separator',  # Strategy for splitting reasoning steps; options are separator, keywords, llm
    judge_config=judge_config,
)
model_config5 = dict(
    report_path='/home/slurm/tuenv2/open_i1_project/evaluation/evalscope/final_benchmark_results/rl/r1_32k/20250325_043943/1',  # Path to the model reasoning results from the previous step
    model_name='DeepSeek-R1-Distill-Qwen-1.5B',  # Model name
    tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',  # Path to the model tokenizer for token count calculation
    dataset_name='math_500',  # Dataset name from the previous step
    subsets=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],  # Subsets from the previous step
    split_strategies='separator',  # Strategy for splitting reasoning steps; options are separator, keywords, llm
    judge_config=judge_config,
)
max_tokens = 200000  # Filter outputs with token counts less than max_tokens to improve evaluation efficiency
count = 200000 
# run_task(model_config5, output_dir='r1_thinking5', max_tokens=max_tokens, count=count, workers=16)
# exit(0)
combine_results([model_config5, model_config4], "out.png", model_names={
    "DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-Distill-Qwen-1.5B",
    "ADuy_8k": "II-Thought-1.5B-Preview",
    "ii-rl-math-code-v1": "II-Math-RL-Phu",
})
max_tokens = 200000  # Filter outputs with token counts less than max_tokens to improve evaluation efficiency
count = 200000  # Filter count outputs for each subset to improve evaluation efficiency
exit(0)
# Evaluate model thinking efficiency
run_task(model_config2, output_dir='r1_thinking3', max_tokens=max_tokens, count=count, workers=16)
