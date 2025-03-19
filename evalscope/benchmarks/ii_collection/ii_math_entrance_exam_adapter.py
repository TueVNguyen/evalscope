from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import OutputType
from evalscope.metrics.math_parser import extract_answer, math_equal, strip_answer_string
from evalscope.utils.logger import get_logger
import os
from evalscope.constants import HubType
# flake8: noqa

logger = get_logger()
# HOME_DIR = os.path.expanduser('~')

@Benchmark.register(
    name='ii_math_entrance_exam',
    pretty_name='II Math Entrance Exam',
    dataset_id='Intelligent-Internet/Math-Bench-final',
    subset_list=['default'],
    metric_list=['AveragePass@1', 'TopK'],
    few_shot_num=0, 
    train_split=None,
    eval_split='train',  # Only train set is available
    prompt_template='{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    dataset_hub=HubType.HUGGINGFACE,
)
class MathGakaoEnAdapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate the prompt for the model input.
        """
        problem = input_d['problem_translation']
        full_prompt = self.prompt_template.format(query=problem)

        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        return strip_answer_string(input_d['final_answer'])

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.
        """
        # Note: Use same extraction method for both of checkpoint/service/custom
        result = strip_answer_string(extract_answer(result))
        return result

    def match(self, gold: str, pred: str) -> float:
        return math_equal(pred, gold)
