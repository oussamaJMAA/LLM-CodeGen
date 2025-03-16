from langchain_ollama.llms import OllamaLLM
import json


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

models = [
    ("DeepSeek Coder", OllamaLLM(model="deepseek-coder:6.7b")),
    ("DeepSeek R1", OllamaLLM(model="deepseek-r1:7b")),
    ("Code Llama", OllamaLLM(model="codellama:7b"))
]

# Function to generate test data dynamically
def generate_test_data(model, user_inputs):
    test_data = {}

    for key, user_input in user_inputs.items():
        prompt = PROMPT_TEMPLATE.format(user_input=user_input)
        response = model.invoke(prompt)
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
    #"threaded_sorting": "Generate a C program that sorts an array using multiple threads (e.g., merge sort with multithreading).",
    # "file_encryption": "Generate a C function to encrypt and decrypt a file using a simple XOR-based cipher.",
    # "dijkstra_algorithm": "Generate a C program that finds the shortest path in a graph using Dijkstra's algorithm.",
    # "memory_pool_allocator": "Generate a C program that implements a custom memory pool allocator for dynamic memory management.",
    # "producer_consumer": "Generate a C program to solve the producer-consumer problem using semaphores and threads.",
    # "LRU_cache": "Generate a C program to implement an LRU (Least Recently Used) cache using a doubly linked list and a hash map.",
    # "http_server": "Generate a basic HTTP server in C using sockets to handle simple GET requests.",
    # "regex_parser": "Generate a C function to implement a basic regular expression parser that supports simple pattern matching."
}

# Generate test data for each model
results = {}
for model_name, model in models:
    print(f"using {model_name} ...")
    results[model_name] = generate_test_data(model, user_inputs)

# Convert results to JSON
json_output = json.dumps(results, indent=4)
#Save the JSON Output
with open("model_test_results.json", "w") as json_file:
    json_file.write(json_output)

print("JSON output saved to model_test_results.json")

