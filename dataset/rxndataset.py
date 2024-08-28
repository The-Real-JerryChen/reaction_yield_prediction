import torch
from torch.utils.data import Dataset
from .chemdata import *
from .utils import *
import torch
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("laituan245/molt5-base")


class ReactionDataset(Dataset):
    def __init__(self, data):
        self.data = self.build_reaction_dataset(data)

    def build_reaction_dataset(self, data):
        reaction_dataset = []
        for reaction_smiles, yield_data in zip(data['rxn'], data['yld']):
            reactants, products = parse_reaction(reaction_smiles)
            reaction = Reaction(reactants, products,yield_data)
            reaction_dataset.append(reaction)
        return reaction_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        reaction = self.data[idx]
        reactants = reaction.reactants
        products = reaction.products
        yield_data = torch.tensor(reaction.yield_data, dtype=torch.float)
        return reactants, products, yield_data
    def calculate_yield_stats(self, indices=None):
        if indices is None:
            indices = range(len(self))  
        
        yield_data_list = [self.data[i].yield_data for i in indices]

        yield_data_tensor = torch.tensor(yield_data_list)

        yield_mean = torch.mean(yield_data_tensor)
        yield_std = torch.std(yield_data_tensor)

        return yield_mean, yield_std

from torch.utils.data import DataLoader

def collate_fn(batch):
    combined_smiles_sequences = []
    combined_graph_sequences = []
    fingerprint_sequences = []
    yield_data = []
    
    for reactants, products, yld in batch:
        reactant_smiles = ".".join([mol.smiles for mol in reactants])
        product_smiles = ".".join([mol.smiles for mol in products])
        combined_smiles = f"{reactant_smiles}>>{product_smiles}"
        combined_smiles_sequences.append(combined_smiles)
        
        reactant_graphs = [mol.graph for mol in reactants if mol.graph is not None]
        product_graphs = [mol.graph for mol in products if mol.graph is not None]
        
        combined_reactant_graph = Batch.from_data_list(reactant_graphs) if reactant_graphs else None
        combined_product_graph = Batch.from_data_list(product_graphs) if product_graphs else None
        
        combined_graph_sequences.append((combined_reactant_graph, combined_product_graph))
        
        fingerprint_seq = torch.stack([convert_fp_to_tensor(mol.fp) for mol in reactants + products], dim=0)
        fingerprint_seq = torch.sum(fingerprint_seq, dim=0)

        fingerprint_sequences.append(fingerprint_seq)
        
        yield_data.append(yld)
    
    tokenized_smiles = tokenizer(combined_smiles_sequences, padding=True, return_tensors='pt')
    
    yield_data = torch.stack(yield_data)
    fingerprint_sequences = torch.stack(fingerprint_sequences)
    return tokenized_smiles, combined_graph_sequences, fingerprint_sequences, yield_data

