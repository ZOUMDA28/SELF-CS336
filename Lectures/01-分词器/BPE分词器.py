import regex
from collections import Counter

DEEPSEEK_REGEX=r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"
def split_graphemes(token):
    return tuple(regex.findall(r'\X',token))
def train_bpe(texts,num_merges=50):
      """
    texts: 文本列表（用于训练BPE）
    num_merges: BPE 迭代合并的次数
    """
      