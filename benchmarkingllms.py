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
from langchain_ollama.llms import OllamaLLM
from tabulate import tabulate
import json

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

# Initialize models
custom_model = ChatBedrockConverse(
    region_name="us-east-1",
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0.2,
)

aws_bedrock = AWSBedrock(model=custom_model)

# Define metrics
metrics = [
    GEval(
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
    ),
 
   GEval(
        model=aws_bedrock,
        name="CyclomaticComplexity",
        criteria="Evaluate the cyclomatic complexity of the generated code.",
        evaluation_steps=[
            "Calculate the cyclomatic complexity based on the control flow graph.",
            "Ensure the complexity is within acceptable limits for maintainability.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        async_mode=False,
    ),
]

print("Starting code generation .....")
# Initialize the models to evaluate
models = [
    ("DeepSeek Coder", OllamaLLM(model="deepseek-coder:6.7b")),
    ("DeepSeek R1", OllamaLLM(model="deepseek-r1:7b")),
    ("Code Llama", OllamaLLM(model="codellama:7b"))
]

# Prompt template
PROMPT_TEMPLATE = """
You are an expert C programmer specializing in embedded systems development with strict compliance to MISRA C coding standards and best practices.
Your task is to generate highly optimized and well-structured C code based on the user's request while ensuring the following rules:

### General Coding Rules for Embedded Systems
- **Memory Management**: Avoid dynamic memory allocation (no malloc/free). Use stack-based or static memory allocation only.
- **Code Standards**: Follow C99 standards and embedded system best practices. Adhere to MISRA C guidelines.
- **Optimization Techniques**: Implement algorithmic optimizations (e.g., loop unrolling, tiling). Utilize hardware features when available (FPU, SIMD instructions). Optimize memory management within embedded constraints.
- **Code Structure**: Maintain modular design with clear separation of concerns. Use .h files for declarations and .c files for implementations. Functions should be well-defined and maintainable.
- **Documentation and Readability**: Use meaningful variable/function names. Include clear comments for complex logic but avoid excessive trivial comments. Document function purpose, parameters, and return values.
- **Testing**: Include test functions to validate correctness. Design tests for easy modification of parameters and generate appropriate test datasets.
- **Portability**: Ensure compatibility with target compilers (e.g., GCC). Make constants/parameters configurable. Write code that is easily portable and maintainable.
- **Efficiency & Safety**:
  - Avoid complex flow constructs (goto, recursion).
  - Ensure all loops have fixed bounds to prevent runaway code.
  - Restrict functions to a single printed page.
  - Use runtime assertions for error handling.
  - Minimize the use of global variables and limit pointer dereferencing.
  - Validate all function return values or explicitly cast to void.
  - Limit function parameters to at most 4.
  - Use `const` qualifiers as much as possible.
  - Maintain cyclomatic complexity < 20.
  - Avoid implicit casts and magic numbers.
  - Use ANSI standard data types from stdint.h (uint8_t, uint16_t, etc.).
  - Do not use `extern` variables in source files; instead, include the proper header file.
  - Avoid unions due to alignment differences.
  - Enclose all debug-related code under conditional compilation.
  - Avoid X-macros and complex macros.
  - Always use braces for blocks and parentheses `()` for expressions.
  - Explicitly write comparisons with zero in conditional expressions.

### User Request
Generate MISRA C-compliant embedded systems code based on the following requirements:

{user_input}

### Output Format
Provide:
1. **MISRA C-Compliant Code** with proper structure, comments, and formatting.

2. **Brief Explanation** of the generated code, highlighting key considerations and optimizations made.
3. **Test Code** if applicable, following best practices for embedded system testing.

Ensure that the code is concise, efficient, and adheres to all the listed guidelines. Do not include unnecessary explanations or unrelated details.
"""

# Function to generate test data dynamically
def generate_test_data(model, user_inputs):
    test_data = {}

    for key, user_input in user_inputs.items():
        prompt = PROMPT_TEMPLATE.format(user_input=user_input)
        response = model(prompt)
        response = response.replace("<think>", "").replace("</think>", "").strip()

        test_data[key] = {
            "input": user_input,
            "actual_output": response
        }

    return test_data

# Example user inputs
user_inputs = {
    "matrix_multiplication": "Generate a C function to perform matrix multiplication for two dynamically allocated matrices.",
    "binary_search_tree": "Generate a C program that implements a binary search tree with insertion, deletion, and in-order traversal.",
    "threaded_sorting": "Generate a C program that sorts an array using multiple threads (e.g., merge sort with multithreading).",
    "file_encryption": "Generate a C function to encrypt and decrypt a file using a simple XOR-based cipher.",
    "dijkstra_algorithm": "Generate a C program that finds the shortest path in a graph using Dijkstra's algorithm.",
    "memory_pool_allocator": "Generate a C program that implements a custom memory pool allocator for dynamic memory management.",
    "producer_consumer": "Generate a C program to solve the producer-consumer problem using semaphores and threads.",
    "LRU_cache": "Generate a C program to implement an LRU (Least Recently Used) cache using a doubly linked list and a hash map.",
    "http_server": "Generate a basic HTTP server in C using sockets to handle simple GET requests.",
    "regex_parser": "Generate a C function to implement a basic regular expression parser that supports simple pattern matching."
}


# Function to evaluate a single test case
def evaluate_test_case(test_case, metrics):
    scores = {}
    for metric in metrics:
        metric.measure(test_case)
        scores[metric.name] = metric.score
    return scores

# Prepare data for the table
table_data = [["Model", "Metric", "Score"]]

for model_name, model in models:
    print(f"Evaluating {model_name}...")

    # Generate test data
    test_data = generate_test_data(model, user_inputs)
    print(json.dumps(test_data, indent=4))
    print("Finished code generation")
    print("*"*200)

    test_cases = []

    # Iterate over the dictionary to create test cases
    for key, data in test_data.items():
        test_case = LLMTestCase(
            input=data["input"],
            actual_output=data["actual_output"]
        )
        test_cases.append(test_case)

    # Use ThreadPoolExecutor for asynchronous evaluation
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda test_case: evaluate_test_case(test_case, metrics), test_cases))

    # Calculate the overall scores as the average
    overall_scores = {metric.name: sum(result[metric.name] for result in results) / len(results) for metric in metrics}

    # Append results to the table
    for metric_name, score in overall_scores.items():
        table_data.append([model_name, metric_name, score])

# Print the table
print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
