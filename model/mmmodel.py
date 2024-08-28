import torch
import torch.nn as nn
from .encoders import *
import numpy as np
class CLME(nn.Module):
    def __init__(self, 
                node_in_feats, edge_in_feats, g_hidden_size, num_step_mp, num_step_set2set, num_layer_set2set, g_output_dim,
                vocab_size, s_embed_dim, num_heads, num_layers, context_length, s_output_dim, cl_hidden_dim = 256
                ):
        super(CLME, self).__init__()

        self.mpnn = MolEncoder(node_in_feats, edge_in_feats,  hidden_size = g_hidden_size, num_step_mp = num_step_mp, num_step_set2set = num_step_set2set, num_layer_set2set = num_layer_set2set,
                 output_dim = g_output_dim)
        self.transformer = SMILES_Encoder(vocab_size = vocab_size, embed_dim= s_embed_dim,  num_heads = num_heads, num_layers = num_layers, context_length = context_length, output_dim = s_output_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gproj = nn.Linear(g_output_dim*2, cl_hidden_dim)
        self.sproj = nn.Linear(s_output_dim, cl_hidden_dim)

        
    def process_graph(self, graph_ls):
        reactant_features = []
        product_features = []

        for reactant_batch, product_batch in graph_ls:
            encoded_reactants = self.mpnn(reactant_batch)
            encoded_products = self.mpnn(product_batch)
            reactant_feature = encoded_reactants.sum(dim=0)
            product_feature = encoded_products.sum(dim=0)
            reactant_features.append(reactant_feature)
            product_features.append(product_feature)

        reactant_features = torch.stack(reactant_features)
        product_features = torch.stack(product_features)
        
        return torch.cat([reactant_features, product_features], 1)

    def forward(self, smiles, graphs):
        smiles_features = self.transformer(smiles)

        graph_features =  self.process_graph(graphs)

        graph_features = self.gproj(graph_features)
        smiles_features = self.sproj(smiles_features)

        smiles_features = smiles_features / smiles_features.norm(dim=1, keepdim=True)
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
        return smiles_features, graph_features, self.logit_scale.exp()



class UAM(nn.Module):
    def __init__(self, 
                mlp_input_size, node_in_feats, edge_in_feats, 
                mlp_hidden_size, dense_l, spar_l, num_exps, mlp_drop, mlp_out_size,
                g_hidden_size, num_step_mp, num_step_set2set, num_layer_set2set, g_output_dim,
                vocab_size, s_embed_dim, num_heads, num_layers, context_length, s_output_dim, 
                predict_hidden_dim, prob_dropout, pre_trained = 'False'):
        super(UAM, self).__init__()

        self.mlp = Fea_Encoder(input_size =mlp_input_size , hidden_size=mlp_hidden_size, dense_layers = dense_l, sparse_layers = spar_l, num_experts = num_exps, \
                 dropout = mlp_drop, output_dim =  mlp_out_size)
        
        self.clme = CLME(node_in_feats, edge_in_feats,g_hidden_size = g_hidden_size, num_step_mp = num_step_mp, num_step_set2set = num_step_set2set, num_layer_set2set = num_layer_set2set,
        g_output_dim = g_output_dim, vocab_size = vocab_size, s_embed_dim = s_embed_dim, num_heads = num_heads, num_layers = num_layers, context_length = context_length, s_output_dim = s_output_dim)
        
        if pre_trained!= 'False':
            self.clme.load_state_dict(torch.load(pre_trained))
        else: 
            print('No pretrained encoder is loaded')


        self.predict = nn.Sequential(
            nn.Linear(g_output_dim * 2 + mlp_out_size +s_output_dim, predict_hidden_dim), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_dim, predict_hidden_dim), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_dim, 2)
        )
    
    def forward(self, smiles, mols, input_feats):
        graph_feats = self.clme.process_graph(mols) 
        feats, a_loss = self.mlp(input_feats)
        seq_feats = self.clme.transformer(smiles)
        concat_feats = torch.cat([ graph_feats, feats, seq_feats], 1)
        out = self.predict(concat_feats)
        mean = out[:,0]
        logvar = out[:,1]
        return mean, logvar, a_loss