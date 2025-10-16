from transformers import AutoTokenizer

# Use the same model name you used for training
model_name = "distilbert-base-uncased"
save_path = "models/hf_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

print("✅ Tokenizer files saved to", save_path)
