"""
Implement Attention.
Ref: https://zhuanlan.zhihu.com/p/366592542
"""


import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, q, k, v):
        U = torch.bmm(q, k.transpose(1, 2))
        U /= self.scale

        attn_map = self.softmax(U)
        output = torch.bmm(attn_map, v)

        return output, attn_map


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_q, d_k, d_v, d_out=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_head = n_head
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_q, n_head * d_q)
        self.fc_k = nn.Linear(d_k, n_head * d_k)
        self.fc_v = nn.Linear(d_v, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=d_k ** 0.5)

        if d_out == None:
            d_out = d_q
        self.fc_out = nn.Linear(n_head * d_v, d_out)
    
    def forward(self, q, k, v):
        n_head, d_q, d_k, d_v = self.n_head, self.d_q, self.d_k, self.d_v
        batch, n_q, d_q = q.size()
        batch, n_k, d_k = k.size()
        batch, n_v, d_v = v.size()

        ## single-head to multi-head
        # (b, n, d) -> (b, n, m x d)
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        # (b, n, m x d) -> (m x b, n, d)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)
        
        ## attention
        attn, attn_map = self.attention(q, k, v)

        ## resize & output
        # (m x b, n, d) -> (b, n, m x d)
        attn = attn.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)
        return self.fc_out(attn)


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_hidden, d_out=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # [n, d_in] x [d_in, d_hidden] = [n, d_hidden]
        self.w_q = nn.Parameter(torch.Tensor(size=(d_in, d_hidden)))
        self.w_k = nn.Parameter(torch.Tensor(size=(d_in, d_hidden)))
        self.w_v = nn.Parameter(torch.Tensor(size=(d_in, d_hidden)))

        self.multi_attn = MultiHeadAttention(n_head=8, d_q=d_hidden, d_k=d_hidden, d_v=d_hidden, d_out=d_out)
        self.init_parameters()
    
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / param.size(-1) ** 0.5
            param.data.uniform_(-stdv, stdv)
    
    def forward(self, hidden_states):
        # (b, n, d_in) -> (b, n, d_hidden)
        q = torch.matmul(hidden_states, self.w_q)
        k = torch.matmul(hidden_states, self.w_k)
        v = torch.matmul(hidden_states, self.w_v)

        attn = self.multi_attn(q, k, v)

        return attn