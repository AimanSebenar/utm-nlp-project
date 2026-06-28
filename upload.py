from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Path to your local retrained model folder
local_path = "C:/Users/Aiman/.vscode/nlp-project/models/roberta"

# 2. Load the model and tokenizer locally
model = AutoModelForSequenceClassification.from_pretrained(local_path)
tokenizer = AutoTokenizer.from_pretrained(local_path)

# 3. Push to HF Hub (Replace with your HF username and desired repo name)
# Set private=True if you do not want anyone else to access your model
repo_id = "AimanSebenar/my-custom-roberta"

model.push_to_hub(repo_id, private=False)
tokenizer.push_to_hub(repo_id, private=False)

print(f"Success! Model uploaded to https://huggingface.co{repo_id}")
