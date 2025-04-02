from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import OutputType
from evalscope.metrics.math_parser import extract_answer, math_equal, strip_answer_string, get_last_boxed, deepscaler_verify, math_verify_boxed
from evalscope.utils.logger import get_logger
from evalscope.constants import HubType
# flake8: noqa

logger = get_logger()
# HOME_DIR = os.path.expanduser('~')

@Benchmark.register(
    name='olympiad_bench',
    pretty_name='Olympiad Bench',
    dataset_id='tuenguyen/eval_math_olympiadbench',
    subset_list=['default'],
    metric_list=['AveragePass@1', 'TopK'],
    few_shot_num=0,
    train_split=None,
    eval_split='train',  # Only train set is available
    prompt_template='{query}\n\nPlease reason step by step, and put your final answer within \\\\boxed{{}}.',
    dataset_hub=HubType.HUGGINGFACE,
)
class OlympiadBenchAdapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate the prompt for the model input.
        """
        problem = input_d['problem']
        full_prompt = self.prompt_template.format(query=problem)

        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        answer = input_d['answer']
        if len(answer) >= 3 and answer[0] == '$' and answer[-1] == '$':
            answer = answer[1:-1]
        return answer

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.
        """
        # Note: Use same extraction method for both of checkpoint/service/custom
        extract_boxed_result = get_last_boxed(result) 
        if extract_boxed_result is  None:
            print(f"no boxed result for {result[-50:]}\n--------------------------------")
            return ""
        return extract_boxed_result

    def match(self, gold: str, pred: str) -> float:
        correct = 0 
        math_verify_result = 0 
        deepscaler_result = 0 
        try:
            # import ipdb; ipdb.set_trace()
            if len(pred.strip()) == 0:
                return 0 
            try:
                math_verify_result = math_verify_boxed(pred, gold) 
                
            except Exception as e:
                print(f"Error matching math_verify_boxed {pred} and {gold}: {e}")
                pass
            try:
                deepscaler_result = deepscaler_verify(pred, gold)
        
            except Exception as e:
                print(f"Error matching deepscaler_verify {pred} and {gold}: {e}")
                pass
            correct = math_equal(pred, gold)
        except Exception as e:
            logger.error(f"Error matching {pred} and {gold}: {e}")
        if int(math_verify_result) + int(deepscaler_result) + int(correct) not in [0, 3]:
            logger.warning(f"math_verify_result {math_verify_result} deepscaler_result {deepscaler_result} correct {correct}: {pred} {gold}")
        # logger.info(f"{math_verify_result} {deepscaler_result} {correct}")
        return int(int(correct) + int(math_verify_result) + int(deepscaler_result) > 0 )
