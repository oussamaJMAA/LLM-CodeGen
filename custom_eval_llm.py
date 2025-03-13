from deepeval import evaluate
from langchain_aws import ChatBedrockConverse
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from deepeval.metrics import GEval, HallucinationMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
import boto3


_ = load_dotenv()
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")


class AWSBedrock(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom AWS Bedrock Model"


custom_model = ChatBedrockConverse(
    region_name="us-east-1",
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0.2,
)

aws_bedrock = AWSBedrock(model=custom_model)
# print(aws_bedrock.generate("Write me a joke"))


correctness_metric = GEval(
    model=aws_bedrock,
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        #LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    # verbose_mode=True
    # async_mode=True
)

test_case = LLMTestCase(
    input="The dog chased the cat up the tree, who ran up the tree?",
    actual_output="the cat for sure since he is faster at climbing trees",
    #expected_output="The cat.",
)
#evaluate(test_cases=[test_case], metrics=[correctness_metric])
# correctness_metric.measure(test_case)
# print(correctness_metric.score, correctness_metric.reason)


# context=["A man with blond-hair, and a brown shirt drinking out of a public water fountain."]

# # Replace this with the actual output from your LLM application
# actual_output="A blond drinking water in public."

# test_case = LLMTestCase(
#     input="What was the blond doing?",
#     actual_output=actual_output,
#     context=context
# )
# metric = HallucinationMetric(model=aws_bedrock, threshold=0.5)

# # To run metric as a standalone
# # metric.measure(test_case)
# # print(metric.score, metric.reason)

# evaluate(test_cases=[test_case], metrics=[metric])






# Define the custom metric for code quality
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
)

# Define a test case with the generated code
test_case = LLMTestCase(
    input="Generate a C function to calculate the factorial of a number following embedded system best practices.",
    actual_output="""
    #include <stdint.h>
    #include <assert.h>

    // Function to calculate factorial
    uint32_t factorial(uint32_t n) {
        uint32_t result = 1;
        for (uint32_t i = 1; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    // Test function
    void test_factorial() {
        assert(factorial(0) == 1);
        assert(factorial(1) == 1);
        assert(factorial(5) == 120);
    }
    """
)

# Measure the code quality
code_quality_metric.measure(test_case)
print("Score:", code_quality_metric.score)
print("Reason:", code_quality_metric.reason)
