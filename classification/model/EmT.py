import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight)
        if self.bias is not None:
            output = output - self.bias
        return F.relu(torch.matmul(adj, output))


class GCN(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm_adj(self, adj):
        rowsum = torch.sum(adj, dim=-1)
        rowsum = rowsum + (rowsum == 0).float()
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def forward(self, data):
        graph, adj = data
        adj = self.norm_adj(adj)
        support = torch.matmul(graph, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return F.relu(output), adj


class ChebyNet(Module):
    def __init__(self, k_order, in_feature, out_feature):
        super().__init__()
        self.K = k_order
        self.filter_weight, self.filter_bias = self.init_filter(k_order, in_feature, out_feature)

    def init_filter(self, k_order, feature, out, bias=True):
        weight = nn.Parameter(torch.FloatTensor(k_order, 1, feature, out), requires_grad=True)
        nn.init.normal_(weight, 0, 0.1)
        bias_ = None
        if bias:
            bias_ = nn.Parameter(torch.zeros((1, 1, out), dtype=torch.float32), requires_grad=True)
            nn.init.normal_(bias_, 0, 0.1)
        return weight, bias_

    @staticmethod
    def get_laplacian(adj):
        degree = torch.sum(adj, dim=1)
        degree_norm = torch.div(1.0, torch.sqrt(degree) + 1.0e-5)
        degree_matrix = torch.diag(degree_norm)
        return -torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)

    def chebyshev(self, x, laplacian):
        x1 = torch.matmul(laplacian, x)
        x_ = torch.stack((x, x1), dim=1)
        if self.K > 1:
            for _ in range(2, self.K):
                x_current = 2 * torch.matmul(laplacian, x_[:, -1]) - x_[:, -2]
                x_ = torch.cat((x_, x_current.unsqueeze(dim=1)), dim=1)

        x_ = x_.permute(1, 0, 2, 3)
        out = torch.matmul(x_, self.filter_weight)
        out = torch.sum(out, dim=0)
        return F.relu(out + self.filter_bias)

    def forward(self, data):
        x, adj = data
        laplacian = self.get_laplacian(adj)
        out = self.chebyshev(x, laplacian)
        return out, adj


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GraphEncoder(nn.Module):
    def __init__(self, num_layers, num_node, in_features, out_features, k_order, graph2token="Linear", encoder_type="GCN"):
        super().__init__()
        self.graph2token = graph2token
        self.K = k_order
        assert graph2token in ["Linear", "AvgPool", "MaxPool", "Flatten"], "graph2vector type is not supported!"
        self.tokenizer = nn.Linear(num_node * out_features, out_features) if graph2token == "Linear" else None
        layers = []
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layer = self.get_layer(encoder_type, in_features, out_features)
            else:
                layer = self.get_layer(encoder_type, out_features, out_features)
            layers.append(layer)
        self.encoder = nn.Sequential(*layers)

    def get_layer(self, encoder_type, in_features, out_features):
        assert encoder_type in ["Cheby", "GCN"], "encoder type is not supported!"
        if encoder_type == "GCN":
            return GCN(in_features, out_features)
        return ChebyNet(self.K, in_features, out_features)

    def forward(self, x, adj):
        output = self.encoder((x, adj))
        x, _ = output
        if self.tokenizer is not None:
            return self.tokenizer(torch.flatten(x, start_dim=1))
        if self.graph2token == "AvgPool":
            return torch.mean(x, dim=-1)
        if self.graph2token == "MaxPool":
            return torch.max(x, dim=-1)[0]
        return torch.flatten(x, start_dim=1)


class MultiScaleSTA(nn.Module):
    def __init__(self, heads, kernel_sizes, dropout=0.0):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        heads,
                        heads,
                        (kernel_size, 1),
                        stride=1,
                        padding=(int(0.5 * (kernel_size - 1)), 0),
                    )
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale_logits = nn.Parameter(torch.zeros(len(self.kernel_sizes)))

    def forward(self, x):
        outputs = [conv(self.dropout(x)) for conv in self.convs]
        if len(outputs) == 1:
            return outputs[0]
        weights = torch.softmax(self.scale_logits, dim=0)
        mixed = torch.stack(outputs, dim=0)
        return torch.sum(weights.view(-1, 1, 1, 1, 1) * mixed, dim=0)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_sizes=(3,), dropout=0.0, alpha=0.25):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.sta = MultiScaleSTA(heads, kernel_sizes, dropout=alpha * dropout)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = self.sta(out)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, kernel_sizes=(3,), dropout=0.0, alpha=0.25):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                kernel_sizes=kernel_sizes,
                                dropout=dropout,
                                alpha=alpha,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimAM1D(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        seq_len = x_permuted.size(2)
        n = max(seq_len - 1, 1)
        x_minus_mu_square = (x_permuted - x_permuted.mean(dim=2, keepdim=True)).pow(2)
        energy = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=2, keepdim=True) / n + self.e_lambda)) + 0.5
        return (x_permuted * self.activation(energy)).permute(0, 2, 1)


class AdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, tokens):
        weights = torch.softmax(self.gate(tokens), dim=1)
        return torch.sum(tokens * weights, dim=1)


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class EmT(nn.Module):
    def __init__(
        self,
        layers_graph=(1, 2),
        layers_transformer=1,
        num_adj=3,
        num_chan=62,
        num_feature=5,
        hidden_graph=16,
        K=2,
        num_head=8,
        dim_head=16,
        dropout=0.25,
        num_class=3,
        alpha=0.25,
        graph2token="Linear",
        encoder_type="GCN",
        use_simam=True,
        fusion_mode="mean",
        pooling_mode="mean",
        sta_kernel_sizes=(3,),
        adj_sparsity_weight=0.0,
        adj_diversity_weight=0.0,
    ):
        super().__init__()
        self.graph_encoder_type = encoder_type
        self.fusion_mode = fusion_mode
        self.pooling_mode = pooling_mode
        self.use_simam = use_simam
        self.adj_sparsity_weight = adj_sparsity_weight
        self.adj_diversity_weight = adj_diversity_weight

        self.ge1 = GraphEncoder(
            num_layers=layers_graph[0],
            num_node=num_chan,
            in_features=num_feature,
            out_features=hidden_graph,
            k_order=K,
            graph2token=graph2token,
            encoder_type=encoder_type,
        )
        self.ge2 = GraphEncoder(
            num_layers=layers_graph[1],
            num_node=num_chan,
            in_features=num_feature,
            out_features=hidden_graph,
            k_order=K,
            graph2token=graph2token,
            encoder_type=encoder_type,
        )

        self.adjs = nn.Parameter(torch.FloatTensor(num_adj, num_chan, num_chan), requires_grad=True)
        nn.init.xavier_uniform_(self.adjs)

        token_dim = hidden_graph
        if graph2token in ["AvgPool", "MaxPool"]:
            token_dim = num_chan
        if graph2token == "Flatten":
            token_dim = num_chan * hidden_graph

        self.transformer = TTransformer(
            depth=layers_transformer,
            dim=token_dim,
            heads=num_head,
            dim_head=dim_head,
            dropout=dropout,
            mlp_dim=dim_head,
            alpha=alpha,
            kernel_sizes=sta_kernel_sizes,
        )
        self.to_gnn_out = nn.Linear(num_chan * num_feature, token_dim, bias=False)
        self.fusion = AdaptiveFusion(token_dim) if fusion_mode == "adaptive" else None
        self.simam = SimAM1D() if use_simam else nn.Identity()
        self.attention_pool = AttentionPooling(token_dim) if pooling_mode == "attention" else None
        self.mlp = nn.Linear(token_dim, num_class)

    def get_base_adj(self):
        return F.relu(self.adjs + self.adjs.transpose(2, 1))

    def get_adj(self, self_loop=True):
        adj = self.get_base_adj()
        if self_loop:
            eye = torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype).unsqueeze(0)
            adj = adj + eye
        return adj

    def fuse_tokens(self, tokens):
        if self.fusion is not None:
            return self.fusion(tokens)
        return torch.mean(tokens, dim=1)

    def pool_sequence(self, x):
        if self.attention_pool is not None:
            return self.attention_pool(x)
        return torch.mean(x, dim=1)

    def forward(self, x):
        batch_size, seq_len, _, _ = x.size()
        x = rearrange(x, "b s c f -> (b s) c f")
        adjs = self.get_adj(self_loop=self.graph_encoder_type != "Cheby")

        x_skip = self.to_gnn_out(torch.flatten(x, start_dim=1))
        x1 = self.ge1(x, adjs[0])
        x2 = self.ge2(x, adjs[1])
        x = self.fuse_tokens(torch.stack((x_skip, x1, x2), dim=1))

        x = rearrange(x, "(b s) h -> b s h", b=batch_size, s=seq_len)
        x = self.simam(x)
        x = self.transformer(x)
        x = self.pool_sequence(x)
        return self.mlp(x)

    def regularization_terms(self):
        zero = self.adjs.new_tensor(0.0)
        base_adj = self.get_base_adj()
        eye = torch.eye(base_adj.shape[-1], device=base_adj.device, dtype=base_adj.dtype).unsqueeze(0)
        off_diag = base_adj * (1 - eye)

        sparse = off_diag.abs().mean() if self.adj_sparsity_weight > 0 else zero
        diverse = zero
        if self.adj_diversity_weight > 0 and off_diag.shape[0] > 1:
            flat_adj = off_diag.reshape(off_diag.shape[0], -1)
            norm_adj = F.normalize(flat_adj, dim=1)
            pair_losses = []
            for idx in range(norm_adj.shape[0]):
                for jdx in range(idx + 1, norm_adj.shape[0]):
                    pair_losses.append(torch.square(torch.sum(norm_adj[idx] * norm_adj[jdx])))
            if pair_losses:
                diverse = torch.stack(pair_losses).mean()

        total = self.adj_sparsity_weight * sparse + self.adj_diversity_weight * diverse
        return {"sparse": sparse, "diverse": diverse, "total": total}


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    data = torch.ones((16, 8, 62, 7))
    model = EmT(
        layers_graph=[1, 2],
        layers_transformer=4,
        num_adj=2,
        num_chan=62,
        num_feature=7,
        hidden_graph=32,
        K=4,
        num_head=16,
        dim_head=32,
        dropout=0.25,
        num_class=2,
        graph2token="Linear",
        encoder_type="Cheby",
        use_simam=True,
        fusion_mode="adaptive",
        pooling_mode="attention",
        sta_kernel_sizes=[3, 5],
        adj_sparsity_weight=1e-4,
        adj_diversity_weight=1e-3,
    )
    print(model)
    print(count_parameters(model))
    out = model(data)
    print("Done", out.shape)
