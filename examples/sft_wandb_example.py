# pip install "mistralai==0.4.2" "weave==0.50.7"
import os, asyncio, json
from pathlib import Path

import weave

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

DATA_PATH = Path("./data")
NUM_SAMPLES = 100 # num of samples to use for eval
PROJECT_NAME = "finetuning-example"

weave.init(PROJECT_NAME)

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

train_ds = read_jsonl(DATA_PATH / "sft_data_example.jsonl")
val_ds = read_jsonl(DATA_PATH / "sft_data_example.jsonl")[0:NUM_SAMPLES]

@weave.op()
def call_mistral(model: str, messages: list, **kwargs) -> str:
    chat_response = client.chat(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    return json.loads(chat_response.choices[0].message.content)

prompt = """You are an expert to detect factual inconsistencies and hallucinations. You will be given a document and a summary.
- Carefully read the full document and the provided summary.
- Identify Factual Inconsistencies: any statements in the summary that are not supported by or contradict the information in the document.
Factually Inconsistent: If any statement in the summary is not supported by or contradicts the document, label it as 0
Factually Consistent: If all statements in the summary are supported by the document, label it as 1

Highlight or list the specific statements in the summary that are inconsistent.
Provide a brief explanation of why each highlighted statement is inconsistent with the document.

Return in JSON format with `consistency` and a `reason` for the given choice.

Document: 
{premise}
Summary: 
{hypothesis}
"""

def format_prompt(prompt, premise:str, hypothesis:str, cls=ChatMessage):
    messages = [
        cls(
            role="user",
            content = prompt.format(premise=premise, hypothesis=hypothesis)
        )
    ]
    return messages

class MistralModel(weave.Model):
    model: str
    prompt: str
    temperature: float = 0.7

    @weave.op
    def create_messages(self, premise:str, hypothesis:str):
        return format_prompt(self.prompt, premise, hypothesis)
    
    @weave.op
    def predict(self, premise:str, hypothesis:str):
        messages = self.create_messages(premise, hypothesis)
        return call_mistral(self.model, messages, temperature=self.temperature)

def accuracy(model_output, target):
    class_model_output = model_output.get("consistency") if model_output else None
    return {"accuracy": class_model_output == target}

class BinaryMetrics(weave.Scorer):
    class_name: str
    eps: float = 1e-8

    @weave.op()
    def summarize(self, score_rows) -> dict:
        score_rows = [score for score in score_rows if score["correct"] is not None]
        tp = sum([not score["negative"] and score["correct"] for score in score_rows])
        fp = sum([not score["negative"] and not score["correct"] for score in score_rows])
        tn = sum([score["negative"] and score["correct"] for score in score_rows])
        fn = sum([score["negative"] and not score["correct"] for score in score_rows])
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        result = {"f1": f1, "precision": precision, "recall": recall}
        return result
    
    @weave.op()
    def score(self, model_output, target) -> dict:
        class_model_output = model_output.get("consistency") if model_output else None
        correct = class_model_output == target
        negative = target == 0
        return {"correct": correct, "negative": negative}

model_7b = MistralModel(model="open-mistral-7b", prompt=prompt, temperature=0.7)
F1 = BinaryMetrics(class_name="consistency")
evaluation = weave.Evaluation(dataset=val_ds, scorers=[accuracy, F1])
asyncio.run(evaluation.evaluate(model_7b))
    
ft_prompt = """You are an expert to detect factual inconsistencies and hallucinations. You will be given a document and a summary.
- Carefully read the full document and the provided summary.
- Identify Factual Inconsistencies: any statements in the summary that are not supported by or contradict the information in the document.
Factually Inconsistent: If any statement in the summary is not supported by or contradicts the document, label it as 0
Factually Consistent: If all statements in the summary are supported by the document, label it as 1

Return in JSON format with `consistency` for the given choice.

Document: 
{premise}
Summary: 
{hypothesis}
"""

answer = """{{"consistency": {label}}}"""  # <- json schema

def format_ft_prompt(row, cls=dict, with_answer=True):
    premise = row["premise"]
    hypothesis = row["hypothesis"]
    messages = [
        cls(
            role="user",
            content=prompt.format(premise=premise, hypothesis=hypothesis)
        )
    ]
    if with_answer:
        label = row['target']
        messages.append(
            cls(
                role="assistant",
                content=answer.format(label=label)
            )
        )
    return messages

formatted_train_ds = [format_ft_prompt(row) for row in train_ds]
formatted_val_ds = [format_ft_prompt(row) for row in val_ds]

def save_jsonl(ds, path):
    with open(path, "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")

save_jsonl(formatted_train_ds, DATA_PATH / "formatted_train.jsonl")
save_jsonl(formatted_val_ds, DATA_PATH / "formatted_val.jsonl")

import os
from mistralai.client import MistralClient

api_key = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)

with open(DATA_PATH / "formatted_train.jsonl", "rb") as f:
    ds_train = client.files.create(file=("formatted_train.jsonl", f))
with open(DATA_PATH / "formatted_val.jsonl", "rb") as f:
    ds_val = client.files.create(file=("formatted_val.jsonl", f))

from mistralai.models.jobs import TrainingParameters, WandbIntegrationIn

created_jobs = client.jobs.create(
    model = "open-mistral-7b",
    training_files = [ds_train.id],
    validation_files = [ds_val.id],
    hyperparameters = TrainingParameters(
        training_steps=35,
        learning_rate=0.0001,
    ),
    integrations = [
        WandbIntegrationIn(
            project=PROJECT_NAME,
            api_key=os.getenv("WANDB_API_KEY"),
        ).dict()
    ],
)

jobs = client.jobs.list()
retrieved_job = jobs.data[0]
retrieved_job.fine_tuned_model
mistral_7b_ft = MistralModel(prompt=ft_prompt, model=retrieved_job.fine_tuned_model)

evaluation = weave.Evaluation(dataset=val_ds, scorers=[accuracy, F1])
asyncio.run(evaluation.evaluate(mistral_7b_ft))
