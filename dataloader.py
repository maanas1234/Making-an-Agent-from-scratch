from datasets import load_dataset
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

dataset = load_dataset(
    "Skylion007/openwebtext",
    split="train",
    streaming=True
)

subset = dataset.take(100_000)

all_tokens = []

for sample in subset:
    tokens = enc.encode_ordinary(sample["text"])
    all_tokens.extend(tokens)

tokens_np = np.array(all_tokens, dtype=np.uint16)

tokens_np.tofile("openwebtext_tokens.bin")