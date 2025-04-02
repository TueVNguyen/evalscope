# Run All Evaluation Script

## Overview
`run_all.py` is a comprehensive evaluation script designed to assess the performance of language models across multiple mathematical and coding benchmarks. It uses the evalscope framework to run evaluations on various datasets including mathematical olympiad problems, coding challenges, and standardized tests.

## Requirements
- Python 3.x
- Hugging Face account token
- Required packages:
  - evalscope
  - huggingface_hub
  - requests

## Usage
```bash
python run_all.py [arguments]
```

### Command Line Arguments
- `--model_name`: Name/path of the model to evaluate (default: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
- `--work_dir`: Directory for output files (default: "outputs")
- `--api_url`: API endpoint URL (default: "http://0.0.0.0:1234/v1")
- `--max_tokens`: Maximum tokens for generation (default: 32768)
- `--cache_dir`: Directory for caching model responses (optional)
- `--system_prompt`: Custom system prompt (optional)
- `--eval_batch_size`: Batch size for evaluation (default: 512)

## Supported Datasets
The script evaluates models on the following datasets:
1. AIME (American Invitational Mathematics Examination)
2. AMC (American Mathematics Competition)
3. Math 500
4. Minerva Math
5. Math Gakao (English)
6. Olympiad Bench
7. IIT Math Entrance Exam
8. IF Eval
9. GPQA
10. MMLU
11. Live Code Bench

## Configuration
Each dataset is configured with specific generation parameters:
- `max_tokens`: Maximum length of generated responses
- `temperature`: 0.6 (default sampling temperature)
- `top_p`: 0.95 (nucleus sampling parameter)
- `n`: Number of samples per prompt (varies by dataset)
- `seed`: 42 (fixed random seed for reproducibility)

## Output
The script creates:
- A model info JSON file in the specified work directory
- Evaluation results for each dataset
- Cache files (if cache_dir is specified)

## Output Directory Structure
When the script runs, it creates a timestamped output directory (e.g., `20250402_123953` format: YYYYMMDD_HHMMSS) containing the following subdirectories:

### Directory Contents
- `configs/`: Contains configuration files and parameters used during the evaluation run
- `reports/`: Aggregated results and performance metrics across all benchmarks
- `reviews/`: Detailed analysis of model responses, including correctness assessment
- `predictions/`: Raw output from the model for each dataset and prompt
- `logs/`: System logs, error messages, and execution details for debugging

The timestamped directory format ensures that multiple evaluation runs can be stored and compared without overwriting previous results.

## Example
```bash
python run_all.py --model_name "my-model" --work_dir "my_results" --max_tokens 16384
```

## Note
- Requires a valid Hugging Face token for authentication
- For the Live Code Bench dataset, uses 32 worker threads for code execution
- Supports caching of results for faster re-evaluation
