import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(SelfAttention, self).__init__()

        assert d_model%n_head == 0

        dh = int(d_model/n_head)
        self.dk = d_model ** -0.5

        self.linear1 = nn.Linear(d_model, dh)
        self.linear2 = nn.Linear(d_model, dh)
        self.linear3 = nn.Linear(d_model, dh)

        self.attend = nn.Softmax(dim = -1)

    def forward(self, x):
        query = self.linear1(x)
        key = self.linear2(x)
        value = self.linear3(x)

        attn = torch.matmul(query, key.transpose(-1, -2)) * self.dk
        attn = self.attend(attn)

        return torch.matmul(attn, value)

class MultiheadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiheadAttention, self).__init__()

        self.nhead = n_head

        self.multihead = nn.ModuleList([SelfAttention(d_model, n_head) for _ in range(n_head)])
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        head = [self.multihead[i](x) for i in range(self.nhead)]
        x = torch.cat(head, dim = -1)
        x = self.linear(x)

        return x

class Encoder(nn.Module):

    def __init__(self, d_model, n_head, d_ff):
        super(Encoder, self).__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace = True),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))

        return x

class ViT(nn.Module):

    def __init__(self, n, d_model, n_head, d_ff, n_class):
        super(ViT, self).__init__()

        self.n_encoder = n

        self.encoder = nn.ModuleList([Encoder(d_model, n_head, d_ff) for _ in range(n)])
        self.pool = nn.AdaptiveAvgPool2d((1, d_model))
        self.classifier = nn.Linear(d_model, n_class)

    def forward(self, x):
        for i in range(self.n_encoder):
            x = self.encoder[i](x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
