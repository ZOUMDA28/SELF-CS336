# 字节级Tokenizer
from collections import Counter
class ByteTokenizer:
 def __init__(self):
     self.vocab_size = 256

 def encode(self, text: str):
     return list(text.encode("utf-8"))

 def decode(self, indices):
     return bytes(indices).decode("utf-8")

# 字符级Tokenizer
class CharTokenizer:
 def __init__(self):
     self.vocab = {}
     self.inverse_vocab = {}

 def encode(self, text: str):
     tokens = []
     for ch in text:
         if ch not in self.vocab:
             idx = len(self.vocab)
             self.vocab[ch] = idx
             self.inverse_vocab[idx] = ch
         tokens.append(self.vocab[ch])
     return tokens

 def decode(self, indices):
     return "".join(self.inverse_vocab[i] for i in indices)

# 计算压缩率（byte/token）
def get_compression_ratio(text: str, token_len: int):
 input_byte_len = len(text.encode("utf-8"))
 return input_byte_len / token_len if token_len > 0 else 1


# 简易 BPE Tokenizer
class BPETokenizer:
 def __init__(self, num_merges):
     self.num_merges = num_merges
     self.merges = {}  # {(a,b): new_token_id}
     self.vocab_size = 256  # 从byte开始

 def get_stats(self, tokens):
     pairs = Counter()
     for i in range(len(tokens) - 1):
         pairs[(tokens[i], tokens[i+1])] += 1
     return pairs

 def merge_tokens(self, tokens, pair, new_token):
     i = 0
     new_tokens = []
     while i < len(tokens):
         if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
             new_tokens.append(new_token)
             i += 2
         else:
             new_tokens.append(tokens[i])
             i += 1
     return new_tokens