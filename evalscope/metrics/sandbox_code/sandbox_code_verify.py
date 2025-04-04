

import requests
from typing import List, Dict, Optional
import re
import os
from sandbox_fusion import (
    run_concurrent,
    submit,
    SubmitRequest,
    TestConfig,
    set_endpoint,
)



def get_code_block(text: str, language: str) -> Optional[str]:
    """Extract the code block in the given language from the text.

    Args:
        text (str): The text to check
        language (str): The language of the code blocks to check for

    Returns:
        str: The code block in the given language, or None if not found
    """
    pattern = rf"```{language}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)

    return match.group(1) if match else None

def verify_sandbox_code(llm_output: str, verification_info: List[Dict[str, str]], client_timeout: float = 10, max_attempts: int = 1, concurrency: int = 128) -> bool:
    """
    Verify the sandbox code with the test case
    """
    endpoint = os.environ.get("SANDBOX_ENDPOINT")
    if not endpoint:
        raise ValueError("SANDBOX_ENDPOINT is not set")
    set_endpoint(endpoint)
    test_cases = verification_info["answer"]["test_cases"]
    language = verification_info["answer"].get("language", "python")
    test_config = TestConfig(
        language=language,
        test_cases=test_cases,
    )
    if not get_code_block(llm_output, language):
        return {
            "score": 0.,
            "test_cases": [
                {
                    "score": 0,
                    "reason": "No code block found",
                }
                for _ in test_cases
            ],
            "reason": "No code block found",
        }
    fmt_test_cases = [
            {"input": {"stdin": tc["input"]}, "output": {"stdout": tc["output"]}}
            for tc in test_cases
    ]
    kwargs = []
    for idx, tc in enumerate(fmt_test_cases):
        submit_request = SubmitRequest(
                dataset="custom_dataset",
                id=idx,
                completion=llm_output,
                config=TestConfig(
                    dataset_type="CommonOJDataset",
                    language=language,
                    provided_data={
                        "id": idx,
                        "content": "Optional: Problem description",
                        "test": [tc],
                    },
                ),
            )
        kwargs.append(
                {
                    "request": submit_request,
                    "client_timeout": client_timeout,
                    "max_attempts": max_attempts,
                }
            )
    responses = run_concurrent(
            func=submit,
            kwargs=kwargs,
            concurrency=concurrency,
        )
    num_passed = len([item for item in responses if item.tests[0].passed])
    score_return = {
        "score": num_passed / len(test_cases),
        "test_cases": [
            {
                "score": 1 if item.tests[0].passed else 0,
                "reason": item.tests[0].reason,
            }
            for item in responses
        ],
        "reason": f"success",
    }
    return score_return
