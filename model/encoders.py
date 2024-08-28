import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data
from collections import OrderedDict


class MolEncoder(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, hidden_size,
                 num_step_mp, num_step_set2set, num_layer_set2set,
                 output_dim):
        super(MolEncoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(node_in_feats, hidden_size), nn.ReLU()
        )
        self.num_step_mp = num_step_mp
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, hidden_size * hidden_size)
        )
        self.gnn_layer = NNConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            nn=edge_network,
            aggr='add' 
        )
        self.activation = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.readout = Set2Set(in_channels=hidden_size * 2,
                               processing_steps=num_step_set2set,
                               num_layers=num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_size * 4, output_dim), nn.PReLU()
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        node_feats = self.linear(x.float())
        edge_attr = edge_attr.float()
        hidden_feats = node_feats.unsqueeze(0)
        node_aggr = [node_feats]
        
        for _ in range(self.num_step_mp):
            node_feats = self.activation(self.gnn_layer(node_feats, edge_index, edge_attr)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(node_aggr, batch)
        graph_feats = self.sparsify(readout)
        
        return graph_feats


class SMILES_Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, context_length = 1024, output_dim = 512):
        super(SMILES_Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.pooler = nn.Linear(embed_dim, output_dim)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        for param in self.transformer_encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)

        x = self.token_embedding(input_ids)  
        x = x + self.positional_embedding[:x.size(1), :]
        x = x.permute(1, 0, 2)  
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            attention_mask = ~attention_mask 

        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)

        x = x.permute(1, 0, 2) 
        x = self.ln_final(x)
        x = self.pooler(x[:, 0, :])
        return x  


from mixture_of_experts import MoE

class Experts(nn.Module):
    def __init__(self, dim, num_experts = 6):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim))
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(torch.einsum('end,edh->enh', x, self.w1))
        #out = self.act(self.norm(hidden1))
        return out
    
class SqueezeLayer(nn.Module):
    def __init__(self, fn=True):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        if self.fn:
            tmp =  torch.unsqueeze(x, dim=1)
            return tmp
        else:
            return torch.squeeze(x[0], dim=1), x[1]

class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, dropout):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.MoE = MoE(dim = hidden_size, num_experts = num_experts, experts = Experts(hidden_size, num_experts = num_experts))
        self.s1 = SqueezeLayer()
        self.s2 = SqueezeLayer(fn = False)
        self.act_fn = nn.ReLU()
    def forward(self, x):
        x = self.linear(x)
        raw = x
        x = self.s1(x)
        x = self.MoE(x)
        x, a_loss = self.s2(x)
        x = raw+x
        x = self.act_fn(self.norm(x))
        x = self.dropout(x)
        return x, a_loss

class Fea_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dense_layers, sparse_layers, num_experts,
                 dropout, output_dim =  256):
        super(Fea_Encoder, self).__init__()
        self.input_layer =nn.Sequential( 
                            nn.Linear(input_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(),
                            nn.Dropout(p=dropout)
                            ) 


        self.dense = nn.ModuleList([nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(),
                            nn.Dropout(p=dropout)) for i in range(dense_layers)])

        
        self.sparse = nn.ModuleList(
            [nn.Sequential(MoELayer(hidden_size,num_experts,dropout)) for i in range(sparse_layers)])
        
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.sparse_layers = sparse_layers
        self.dense_layers = dense_layers
        self._initialize_weights()
        self.act =  nn.ReLU()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=1.0)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.input_layer(x)
        loss = 0
        for i in range(self.sparse_layers):
            x, tmp_loss = self.sparse[i](x)
            loss += tmp_loss
        for i in range(self.dense_layers):
            x = self.dense[i](x)
        x = self.output_layer(x)
        return x, loss