import torch
from torch.utils.data import random_split, DataLoader
from model.encoders import *
from model.mmmodel import *
from dataset import ReactionDataset, collate_fn, tokenizer
import pickle
from utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb


config = load_config('./config.yaml')
train_config = config['training']
if 'SM' in train_config['dataset_path']:
    used_dataset = 'SM'
else: 
    used_dataset = 'BH'

 
mlp_config = config['mlp_model']
gnn_config = config['graph_model']
seq_config = config['smiles_model']
predictor_config = config['predictor']

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open(train_config['dataset_path'], 'rb') as f:
    data = pickle.load(f)

vocab_length = tokenizer.vocab_size
reaction_dataset = ReactionDataset(data)
dataset_size = len(reaction_dataset)

train_size = int(0.7 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size 

train_data, val_data, test_data = random_split(reaction_dataset, [train_size, val_size, test_size])
train_mean, train_std = train_data.dataset.calculate_yield_stats(train_data.indices)
print(train_mean,train_std)
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

# wandb.init(
#     project="rxnyld",
# )


trainloader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
valloader = DataLoader(val_data, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn)
testloader = DataLoader(test_data, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn)

model = UAM(1024, 9, 3, mlp_hidden_size = mlp_config['mlp_hidden_size'],
            dense_l = mlp_config['dense_layers'], spar_l = mlp_config['sparse_layers'], num_exps = mlp_config['num_experts'], mlp_drop = mlp_config['dropout_ratio'], mlp_out_size = mlp_config['output_dim'],
            g_hidden_size = gnn_config['hidden_size'], num_step_mp = gnn_config['num_step_mp'], num_step_set2set = gnn_config['num_step_set2set'], num_layer_set2set = gnn_config['num_layer_set2set'], g_output_dim = gnn_config['output_dim'],
            vocab_size = vocab_length, s_embed_dim = seq_config['embed_dim'], num_heads = seq_config['num_heads'], num_layers = seq_config['num_layers'], context_length = seq_config['context_length'], s_output_dim = seq_config['output_dim'], 
            predict_hidden_dim = predictor_config['hidden_size'], prob_dropout = predictor_config['dropout_ratio']).to(device)
criterion = nn.MSELoss(reduction = 'none')

optimizer = optim.AdamW(model.parameters(), lr = train_config['lr'], weight_decay  = train_config['weight_decay'])
lr_scheduler = CosineAnnealingLR(optimizer, T_max=train_config['epochs'], eta_min=5e-5)

best_model = 0.2
for epoch in tqdm(range(train_config['epochs'])):
    trainloss = train(model, trainloader, optimizer, lr_scheduler, criterion, device, train_mean, train_std)
    validr2 = valid(model, valloader, criterion, device, train_mean, train_std)

    if validr2 > best_model:
        best_model = validr2
        torch.save(model.state_dict(), f'./checkpoints/{used_dataset}_model.pth')
    print("Epoch: {} Train Loss: {:.3f} Valid R2: {:.3f}".format(epoch+1, trainloss, validr2))

model.load_state_dict(torch.load(f'./checkpoints/{used_dataset}_model.pth'))
testr2 = test(model, testloader, device, train_mean, train_std)