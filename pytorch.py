import torch
import numpy
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

x = torch.rand(5, 3)
print(x)
print(pipeline('sentiment-analysis')('we love you'))

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
result = generator(
    "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"
)  # doctest: +SKIP
print(result)

sequence = "In a hole in the ground there lived a hobbit."
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer(sequence))

