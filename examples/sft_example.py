import os
from mistralai import Mistral

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

training_data = client.files.upload(
    file={
        "file_name": "sft_data_example.jsonl",
        "content": open("sft_data_example.jsonl", "rb"),
    }
)
validation_data = client.files.upload(
    file={
        "file_name": "sft_data_example.jsonl",
        "content": open("sft_data_example.jsonl", "rb"),
    }
)

created_jobs = client.fine_tuning.jobs.create(
    model = "mistral-small-latest",
    training_files = [training_data.id],
    validation_files = [validation_data.id],
    hyperparameters = {
        "training_steps": 10,
        "learning_rate": 1e-5,
    },
    auto_start = False,
    integrations = [
        {
            "project": "finetuning-example",
            "api_key": WANDB_API_KEY,
        }
    ]
)

client.fine_tuning.jobs.get(job_id=created_jobs.id)
client.fine_tuning.jobs.start(job_id=created_jobs.id)

chat_response = client.chat.complete(
    model=retrieved_job.fine_tuned_model,
    messages=[
        {"role": "user", "content": "Hello"},
    ],
)