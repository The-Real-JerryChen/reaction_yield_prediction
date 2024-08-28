import torch
import numpy as np
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

def mol_to_graph_data_obj_simple(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)  
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)  

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def get_ECFP(smiles, dim = 1024):
    molecule = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=dim)
    return fingerprint.ToBitString()


def parse_reaction(reaction_smiles: str):
    try:
        reactants_smiles, products_smiles = reaction_smiles.split('>>')
        
        reactants =  reactants_smiles.split('.')
        products = products_smiles.split('.')
        
        return reactants, products
    except ValueError:
        raise ValueError("Reaction SMILES should contain '>>' to separate reactants and products.")

def convert_fp_to_tensor(fp_str):
    fp_list = [int(bit) for bit in fp_str]
    return torch.tensor(fp_list, dtype=torch.float)
