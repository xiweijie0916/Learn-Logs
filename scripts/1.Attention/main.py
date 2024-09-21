"""
Implement Attention.
Ref: https://zhuanlan.zhihu.com/p/366592542
"""


import torch
import torch.nn as nn

from AttentionBlocks import ScaledDotProductAttention, SelfAttention, MultiHeadAttention


def attention_example():
    print("=> show attention")
    n_q, n_k, n_v = 2, 4, 4         # n_k must equal n_v
    d_q, d_k, d_v = 128, 128, 64    # d_q must equal d_k
    batch = 32

    q = torch.randn(size=(batch, n_q, d_q))
    k = torch.randn(size=(batch, n_k, d_k))
    v = torch.randn(size=(batch, n_v, d_v))

    Attention = ScaledDotProductAttention(scale=d_k**0.5)
    attn, attn_map = Attention(q, k, v)

    print(attn.size())
    print(attn_map.size())


def multi_head_attention_example():
    print("=> show multi-head attention")
    n_q, n_k, n_v = 2, 4, 4         # n_k must equal n_v
    d_q, d_k, d_v = 128, 128, 64    # d_q must equal d_k
    batch = 32
    n_head = 8

    q = torch.randn(size=(batch, n_q, d_q))
    k = torch.randn(size=(batch, n_k, d_k))
    v = torch.randn(size=(batch, n_v, d_v))

    multi_attn = MultiHeadAttention(n_head=n_head, d_q=d_q, d_k=d_k, d_v=d_v)
    attn = multi_attn(q, k, v)

    print(attn.size())


def self_attention_example():
    print("=> show self-attention")
    b, n, d_in = 32, 64, 128
    hidden_states = torch.randn(size=(b, n, d_in))
    self_attn = SelfAttention(d_in=d_in, d_hidden=128, d_out=256)
    attn = self_attn(hidden_states)

    print(attn.size())


if __name__ == "__main__":
    attention_example()
    multi_head_attention_example()
    self_attention_example()