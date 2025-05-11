import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_path = "./gpt2-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
prompt = "Once upon a time, in a land far away, there lived a kind-hearted dragon."
inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=150,
    num_return_sequences=3,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    pad_token_id=tokenizer.eos_token_id  
)
print("\n=== Generated Stories ===\n")
for i, output in enumerate(outputs):
    story = tokenizer.decode(output, skip_special_tokens=True)
    print(f"--- Story {i+1} ---")
    print(story)
    print()
