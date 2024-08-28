import torch
from torch.utils.data import random_split, DataLoader
from model.encoders import *
from model.mmmodel import *
from dataset import ReactionDataset, collate_fn, tokenizer
import pickle
from utils import *
import torch.optim as optim
from tqdm import tqdm


config = load_config('./config.yaml')
pretrain_config = config['pretraining']
train_config = config['training']
if 'SM' in train_config['dataset_path']:
    used_dataset = 'SM'
else: 
    used_dataset = 'BH'

gnn_config = config['graph_model']
seq_config = config['smiles_model']


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open(train_config['dataset_path'], 'rb') as f:
    data = pickle.load(f)

vocab_length = tokenizer.vocab_size
reaction_dataset = ReactionDataset(data)
dataset_size = len(reaction_dataset)

train_size = int(0.9 * dataset_size)
val_size = int(0.1 * dataset_size)
train_data, val_data= random_split(reaction_dataset, [train_size, val_size])

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
trainloader = DataLoader(train_data, batch_size=pretrain_config['batch_size'], shuffle=True, collate_fn=collate_fn)
valloader = DataLoader(val_data, batch_size=pretrain_config['batch_size'], shuffle=False, collate_fn=collate_fn)

model = CLME(9, 3, g_hidden_size = gnn_config['hidden_size'], num_step_mp = gnn_config['num_step_mp'], num_step_set2set = gnn_config['num_step_set2set'], num_layer_set2set = gnn_config['num_layer_set2set'], g_output_dim = gnn_config['output_dim'],
            vocab_size = vocab_length, s_embed_dim = seq_config['embed_dim'], num_heads = seq_config['num_heads'], num_layers = seq_config['num_layers'], context_length = seq_config['context_length'], s_output_dim = seq_config['output_dim']).to(device)

optimizer = optim.AdamW(model.parameters(), lr = pretrain_config['lr'])


def train(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        smiles_batch, graph_batch, _, _ = batch  
        smiles_batch = smiles_batch.to(device)
        graph_batch = [(rmols.to(device), pmols.to(device)) for rmols, pmols in graph_batch]
        optimizer.zero_grad()

        smiles_features, graph_features, logit_scale = model(smiles_batch, graph_batch)

        loss = do_CL(smiles_features, graph_features, logit_scale) + do_CL(graph_features, smiles_features, logit_scale)

        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item()

    return running_loss / len(dataloader)


def valid(model, dataloader, device):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            smiles_batch, graph_batch, _, _ = batch  
            smiles_batch = smiles_batch.to(device)
            graph_batch = [(rmols.to(device), pmols.to(device)) for rmols, pmols in graph_batch]

            smiles_features, graph_features, logit_scale = model(smiles_batch, graph_batch)

            loss = do_CL(smiles_features, graph_features, logit_scale) + do_CL(graph_features, smiles_features, logit_scale)

            valid_loss += loss.detach().item()

    return valid_loss / len(dataloader)

best_model = 666
for epoch in tqdm(range(pretrain_config['epochs'])):
    trainloss = train(model, trainloader, optimizer, device)
    validloss = valid(model, valloader,  device)

    if validloss < best_model:
        best_model = validloss
        torch.save(model.state_dict(), f'./checkpoints/pretrained.pth')
    print("Epoch: {} Train Loss: {:.3f} Valid Loss: {:.3f}".format(epoch+1, trainloss, validloss))
