import urllib.request
import re
from importlib.metadata import version
import tiktoken
print("Tiktoken version: ", version("tiktoken"))

url = ('https://raw.githubusercontent.com/rasbt/'
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"

urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of characters: ", len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed[:30]))

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer , token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokeneizer = SimpleTokenizerV1(vocab)
text = """It's the last he painted, you know, "Mrs. Gisburn said with pardonable pride."""
ids = tokeneizer.encode(text)
print(ids)
print(tokeneizer.decode(ids))

all_tokens = sorted(list(set(preprocessed)));
all_tokens.extend(["<|endoftext|>", "<|unk|>"]);
vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokeneizer = tiktoken.get_encoding("gpt2")

with open ("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokeneizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:         {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", tokeneizer.decode([desired]))
