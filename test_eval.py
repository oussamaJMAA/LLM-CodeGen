import time
import random
import boto3
from botocore.exceptions import ClientError
from deepeval import evaluate
from langchain_aws import ChatBedrockConverse
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from concurrent.futures import ThreadPoolExecutor

_ = load_dotenv()
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

class AWSBedrock(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return self._make_request_with_backoff(chat_model.invoke, prompt)

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom AWS Bedrock Model"

    def _make_request_with_backoff(self, operation, prompt, max_retries=10):
        retries = 0
        while retries < max_retries:
            try:
                response = operation(prompt)
                return response.content
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    wait_time = min(2 ** retries + random.uniform(0, 1), 60)
                    print(f"ThrottlingException encountered. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    raise e
        raise Exception("Max retries exceeded. API request failed.")

custom_model = ChatBedrockConverse(
    region_name="us-east-1",
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0.2,
)

aws_bedrock = AWSBedrock(model=custom_model)

correctness_metric = GEval(
    model=aws_bedrock,
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)

test_case = LLMTestCase(
    input="The dog chased the cat up the tree, who ran up the tree?",
    actual_output="the cat for sure since he is faster at climbing trees",
)

code_quality_metric = GEval(
    model=aws_bedrock,
    name="CodeQuality",
    criteria="Evaluate the quality of the generated code based on specific rules.",
    evaluation_steps=[
        "Check for dynamic memory allocation (malloc/free) and penalize if found.",
        "Ensure the code adheres to C99 standards and embedded system best practices.",
        "Verify the use of algorithmic optimizations (e.g., loop unrolling, tiling).",
        "Check for modular design with clear separation of concerns.",
        "Ensure meaningful variable/function names and clear comments.",
        "Validate the presence of test functions and appropriate test datasets.",
        "Ensure compatibility with target compilers and configurable constants/parameters.",
        "Avoid complex flow constructs (goto, recursion) and ensure fixed loop bounds.",
        "Restrict functions to a single printed page and use runtime assertions for error handling.",
        "Minimize global variables and pointer dereferencing.",
        "Validate all function return values or explicitly cast to void.",
        "Limit function parameters to at most 4 and use const qualifiers.",
        "Maintain cyclomatic complexity < 20 and avoid implicit casts and magic numbers.",
        "Use ANSI standard data types from stdint.h and avoid extern variables.",
        "Avoid unions and complex macros, and enclose debug code under conditional compilation.",
        "Use braces for blocks and parentheses for expressions, and explicitly write comparisons with zero.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    async_mode=False,
)

test_data = {
    "factorial_function": {
        "input": "Generate a C function to calculate the factorial of a number following embedded system best practices.",
        "actual_output": """
        #include <stdint.h>
        #include <assert.h>

        uint32_t factorial(uint32_t n) {
            uint32_t result = 1;
            for (uint32_t i = 1; i <= n; i++) {
                result *= i;
            }
            return result;
        }

        void test_factorial() {
            assert(factorial(0) == 1);
            assert(factorial(1) == 1);
            assert(factorial(5) == 120);
        }
        """
    },
    "sum_array": {
        "input": "Generate a C function to sum the elements of an array following embedded system best practices.",
        "actual_output": """
        #include <stdint.h>
        #include <assert.h>

        uint32_t sum_array(const uint32_t* array, uint32_t length) {
            uint32_t sum = 0;
            for (uint32_t i = 0; i < length; i++) {
                sum += array[i];
            }
            return sum;
        }

        void test_sum_array() {
            uint32_t array[] = {1, 2, 3, 4, 5};
            assert(sum_array(array, 5) == 15);
        }
        """
    },
    "find_max": {
        "input": "Generate a C function to find the maximum value in an array following embedded system best practices.",
        "actual_output": """
        #include <stdint.h>
        #include <assert.h>
        """
    }
    # Add more test cases as needed
}

test_cases = []

# Iterate over the dictionary to create test cases
for key, data in test_data.items():
    test_case = LLMTestCase(
        input=data["input"],
        actual_output=data["actual_output"]
    )
    test_cases.append(test_case)

# Function to evaluate a single test case
def evaluate_test_case(test_case):
    code_quality_metric.measure(test_case)
    return code_quality_metric.score

# Use ThreadPoolExecutor for asynchronous evaluation
with ThreadPoolExecutor() as executor:
    scores = list(executor.map(evaluate_test_case, test_cases))

# Calculate the overall score as the average
overall_score = sum(scores) / len(scores) if scores else 0

print(f"Overall Score: {overall_score}")
