from deepeval import evaluate
from langchain_aws import ChatBedrockConverse
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
import boto3
_ = load_dotenv()
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

class AWSBedrock(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
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
    model = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0.2
)

aws_bedrock = AWSBedrock(model=custom_model)
#print(aws_bedrock.generate("Write me a joke"))


correctness_metric = GEval(
    model = aws_bedrock,
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    #verbose_mode=True 
    #async_mode=True 
)

test_case = LLMTestCase(
    input="The dog chased the cat up the tree, who ran up the tree?",
    actual_output="the cat for sure since he is faster at climbing trees",
    expected_output="The cat."
)
evaluate(test_cases=[test_case], metrics=[correctness_metric])
# correctness_metric.measure(test_case)
# print(correctness_metric.score, correctness_metric.reason)