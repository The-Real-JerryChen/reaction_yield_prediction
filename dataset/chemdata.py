import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from .utils import *


class Molecule:
    def __init__(self, smiles: str, generate_ECFP = True, generate_graph = True):
        self.smiles = smiles
        self.graph =  None 
        self.fp = None
        if generate_graph:
            self.add_graph()
        if generate_ECFP:
            self.fp = get_ECFP(smiles)
    
    def add_graph(self):
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is not None:
            self.graph = mol_to_graph_data_obj_simple(mol)
        else:
            raise ValueError("Invalid SMILES string.")

    def add_other_information(self, info_key, info_value):
        setattr(self, info_key, info_value)


class Reaction:
    def __init__(self, reactants: list, products: list, yield_data=None):
        self.reactants = [Molecule(smiles) for smiles in reactants]
        self.products = [Molecule(smiles) for smiles in products]
        self.yield_data = yield_data
    
    def add_yield(self, yield_data):
        self.yield_data = yield_data

    def add_reactant(self, smiles):
        self.reactants.append(Molecule(smiles))

    def add_product(self, smiles):
        self.products.append(Molecule(smiles))

    def add_other_information(self, info_key, info_value):
        setattr(self, info_key, info_value)