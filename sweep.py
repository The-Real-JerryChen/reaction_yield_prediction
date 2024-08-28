import torch
from torch.utils.data import random_split, DataLoader
from model.encoders import *
from model.mmmodel import *
from dataset import ReactionDataset, collate_fn, tokenizer
import pickle
from utils import *
import torch.optim as optim
from tqdm import tqdm
import wandb




config = load_config('./config.yaml')
train_config = config['training']
mlp_config = config['mlp_model']
gnn_config = config['graph_model']
seq_config = config['smiles_model']
predictor_config = config['predictor']

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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

print(f"Training set mean {train_mean}, std {train_std}")
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

sweep_config = {
    'method': 'grid',  
    'metric': {
        'name': 'test_r2',  
        'goal': 'maximize'  
    },
    'parameters': {
        'mlp_hidden_size': {
            'values': [256, 512, 1024]
        },
        'mlp_out_size': {
            'values': [512, 1024]
        },
        'mlp_dropout': {
            'values': [0.1, 0.2, 0.3]
        },
        'mlp_num_experts': {
            'values': [4, 5, 6]
        },
        'mlp_dense_layers': {
            'values': [3, 4, 5, 6]
        },
        'mlp_sparse_layers': {
            'values': [3, 4, 5, 6]
        },
        'g_hidden_size': {
            'values': [64, 128]
        },
        'g_output_dim': {
            'values': [512, 1024]
        },
        'num_layer_set2set': {
            'values': [1, 2, 3]
        },
        'num_step_mp': {
            'values': [2, 3]
        },
        's_embed_dim': {
            'values': [256, 512, 1024]
        },
        'num_heads': {
            'values': [4, 6, 8]
        },
        's_output_dim': {
            'values': [512, 1024]
        },
        's_num_layers': {
            'values': [2, 3, 4]
        },
        'predict_hidden_dim': {
            'values': [512, 768]
        },
        'prob_dropout': {
            'values': [0.1, 0.2, 0.3, 0.4]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="rxnyld")


trainloader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
valloader = DataLoader(val_data, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn)
testloader = DataLoader(test_data, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn)

def train_sweep():
    wandb.init()
    model = UAM(1024, 9, 3, 
                mlp_hidden_size=wandb.config.mlp_hidden_size,
                dense_l= wandb.config.mlp_dense_layers,
                spar_l=wandb.config.mlp_sparse_layers, 
                num_exps=wandb.config.mlp_num_experts, 
                mlp_drop=wandb.config.mlp_dropout, 
                mlp_out_size=wandb.config.mlp_out_size,
                g_hidden_size=wandb.config.g_hidden_size, 
                num_step_mp=wandb.config.num_step_mp, 
                num_step_set2set=gnn_config['num_step_set2set'], 
                num_layer_set2set=wandb.config.num_layer_set2set, 
                g_output_dim=wandb.config.g_output_dim,
                vocab_size=vocab_length, 
                s_embed_dim=wandb.config.s_embed_dim, 
                num_heads=wandb.config.num_heads, 
                num_layers=wandb.config.s_num_layers, 
                context_length=seq_config['context_length'], 
                s_output_dim=wandb.config.s_output_dim, 
                predict_hidden_dim=wandb.config.predict_hidden_dim, 
                prob_dropout=wandb.config.prob_dropout).to(device)

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    tolerance = 0
    best_model = 0.2
    for epoch in tqdm(range(train_config['epochs'])):
        trainloss = train(model, trainloader, optimizer, criterion, device, train_mean, train_std)
        validr2 = valid(model, valloader, criterion, device, train_mean, train_std)

        wandb.log({'epoch': epoch+1, 'train_loss': trainloss, 'valid_r2': validr2})

        if validr2 > best_model:
            tolerance = 0
            best_model = validr2
            torch.save(model.state_dict(), './checkpoints/model.pth')
        else:
            tolerance +=1
        print("Epoch: {} Train Loss: {:.3f} Valid R2: {:.3f}".format(epoch+1, trainloss, validr2))
        if tolerance >= 40:
            break


    model.load_state_dict(torch.load('./checkpoints/model.pth'))
    testr2 = test(model, testloader, device, train_mean, train_std)

    wandb.log({'test_r2': testr2})

wandb.agent(sweep_id, train_sweep)
