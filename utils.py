import yaml
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(model, dataloader, optimizer, scheduler, criterion, device, mean, std):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        smiles_batch, graph_batch, fp_batch, labels = batch  
        labels = (labels - mean) / std

        labels = labels.to(device)
        smiles_batch = smiles_batch.to(device)
        graph_batch = [(rmols.to(device), pmols.to(device)) for rmols, pmols in graph_batch]
        fp_batch = fp_batch.to(device)
        optimizer.zero_grad()

        pred, logvar, a_loss = model(smiles_batch, graph_batch, fp_batch)
        # if use kl loss
        # pred2, logvar2, a_loss2 = model(smiles_batch, graph_batch, fp_batch)

        loss = criterion(pred, labels)
        loss = (1 - 0.1) * loss.mean() + 0.1 * ( loss * torch.exp(-logvar) + logvar ).mean() + 0.01 * a_loss.mean() #+ 0.01 * compute_kl_loss(pred,pred2)

        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.detach().item()

    return running_loss / len(dataloader)



def valid(model, dataloader, criterion, device, mean, std):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in dataloader:
            smiles_batch, graph_batch, fp_batch, labels = batch

            y_true.append(labels.cpu().numpy())  
            smiles_batch = smiles_batch.to(device)
            graph_batch = [(rmols.to(device), pmols.to(device)) for rmols, pmols in graph_batch]
            fp_batch = fp_batch.to(device)
            pred, _, _ = model(smiles_batch, graph_batch, fp_batch)
            assert not torch.any(torch.isnan(pred)), "Model output contains NaN values!"

            pred = pred.cpu().numpy() * std.numpy() + mean.numpy()
            
            y_pred.append(pred)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)** 0.5
    r2 = r2_score(y_true, y_pred)

    return r2

def test(model, dataloader, device, mean, std):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in dataloader:
            smiles_batch, graph_batch, fp_batch, labels = batch

            y_true.append(labels.cpu().numpy())  
            smiles_batch = smiles_batch.to(device)
            graph_batch = [(rmols.to(device), pmols.to(device)) for rmols, pmols in graph_batch]
            fp_batch = fp_batch.to(device)
            pred, _, _ = model(smiles_batch, graph_batch, fp_batch)
            pred = pred.cpu().numpy() * std.numpy() + mean.numpy()
            
            y_pred.append(pred)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f'MAE: {mae:.3f}, MSE: {mse:.3f}, R2: {r2:.3f}')
    return r2



def do_CL(X, Y, logit_scale):
    criterion = torch.nn.CrossEntropyLoss()  
    B = X.size(0)
    logits = torch.matmul(X, Y.T) * logit_scale
    labels = torch.arange(B, device=logits.device)
    CL_loss = criterion(logits, labels).mean()
    return CL_loss


def compute_kl_loss( p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()
    return (p_loss + q_loss) / 2
