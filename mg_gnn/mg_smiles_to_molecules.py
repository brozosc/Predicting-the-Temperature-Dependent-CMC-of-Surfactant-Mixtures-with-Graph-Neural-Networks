# -*- coding: utf-8 -*-
"""
@author: BrozosCh
The code generates molecular graphs from SMILES strings. It can generate the molecular graphs of up to two SMILES strings simultaneously. 
example use: 
molecule_1 = MyOwnDataset( root = '', molecule = 1)  -> will return the Data object for the SMILES 1
molecule_2 = MyOwnDataset( root = '', molecule = 2)  -> will return the Data object for the SMILES 2
"""

import torch
import pandas as pd
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import Draw
import math
from rdkit.Chem import rdMolDescriptors



class MyOwnDataset(InMemoryDataset):
    
    types = {'C' :0 , 'N' : 1, 'O' : 2, 'S' : 3, 'F' :4, 'Cl' : 5, 'Br' :6, 'Na' : 7, 'I': 8, 'B' :9, 'K' :10, 'H' :11, 'Li' :12} # atom types
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.AROMATIC: 2} # bond types
    
    def __init__(self, root,molecule, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data_1, self.slices_1  = torch.load(self.processed_paths[0])
        self.data_2, self.slices_2  = torch.load(self.processed_paths[1])
        if molecule == 1:
            self.data , self.slices =self.data_1, self.slices_1
        elif molecule == 2: 
            self.data , self.slices =self.data_2, self.slices_2
        else:
            print('Undefined molecule sequence')
    
 

    @property
    def raw_file_names(self):
        return 'raw.csv'

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt']

    def download(self):
        pass

    def get_node_features(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        type_idx = []
        ring = []
        aromatic = []
        sp2 = []
        sp3 = []
        unspecified = []
        cw = []
        ccw = []
        neutral, positive, negative = [], [], []
        num_hs = []
        num_neighbors = []
            
        for atom in mol.GetAtoms():
            type_idx.append(self.types[atom.GetSymbol()])
            ring.append(1 if atom.IsInRing() else 0)
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)     
            unspecified.append(1 if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED else 0)
            cw.append(1 if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW else 0)
            ccw.append(1 if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else 0)
            negative.append(1 if atom.GetFormalCharge() == -1 else 0)   
            neutral.append(1 if atom.GetFormalCharge() == 0 else 0)
            positive.append(1 if atom.GetFormalCharge() == 1 else 0)
            num_neighbors.append(len(atom.GetNeighbors()))
            num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
                         

        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
        x2 = torch.tensor([ring, aromatic, sp2, sp3, unspecified, cw, ccw, negative, neutral, positive], dtype=torch.float).t().contiguous()
        x3 = F.one_hot(torch.tensor(num_neighbors), num_classes=5)
        x4 = F.one_hot(torch.tensor(num_hs), num_classes=5)
        x = torch.cat([x1.to(torch.float), x2, x3.to(torch.float),x4.to(torch.float)], dim=-1)

        return x
    
    def get_edge_features(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        N = mol.GetNumAtoms()
        row, col, bond_idx, conj, ring, stereo = [], [], [], [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_idx += 2 * [self.bonds[bond.GetBondType()]]
            conj.append(bond.GetIsConjugated())
            conj.append(bond.GetIsConjugated())
            ring.append(bond.IsInRing())
            ring.append(bond.IsInRing())
            stereo.append(bond.GetStereo())
            stereo.append(bond.GetStereo())

        edge_index = torch.tensor([row, col], dtype=torch.long)
        e1 = F.one_hot(torch.tensor(bond_idx),num_classes=len(self.bonds)).to(torch.float)
        e2 = torch.tensor([conj, ring], dtype=torch.float).t().contiguous()
        e3 = F.one_hot(torch.tensor(stereo),num_classes=3).to(torch.float)
        edge_attr = torch.cat([e1, e2, e3], dim=-1)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

        return edge_attr, edge_index


    def get_hydrogen_bond(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        h_donors = rdMolDescriptors.CalcNumLipinskiHBD(mol)
        h_acceptors = rdMolDescriptors.CalcNumLipinskiHBA(mol)
     #   h_donors = rdMolDescriptors.CalcNumHBD(mol)
     #   h_acceptors = rdMolDescriptors.CalcNumHBA(mol)
        return h_donors, h_acceptors

    def process(self):


        df = pd.read_csv(self.raw_paths[0], sep = ';')
        data_list_1 = []
        data_list_2 = []

        for _, row in df.iterrows():
            smiles_1, smiles_2, ratio_1, ratio_2, log_CMC, T_norm = row['smiles_string_surfactant_1'], row['smiles_string_surfactant_2'], row['ratio_1'],  row['ratio_2'], row['log_CMC'], row['T_norm']  # T_norm is the normalized temperature between [0,1]. On the GNN model, the normalized temperature is re-scaled between [0,10].
            print(_, smiles_1, smiles_2, ratio_1)
            

            x_1 = self.get_node_features(smiles_1)
            edge_attr_1, edge_index_1 = self.get_edge_features(smiles_1)
            h_donors_1, h_acceptors_1 = self.get_hydrogen_bond(smiles_1)

            if ratio_2 == 0:
                print('Single surfactant')
                x_2, edge_attr_2, edge_index_2  = torch.zeros((1,33), dtype=torch.float),  \
                                                  torch.zeros((1,8), dtype=torch.float),  \
                                                  torch.zeros((2,1), dtype=torch.int)   
                smiles_2 = 'no_molecule' 
                h_donors_2, h_acceptors_2 = 'no_molecule', 'no_molecule' 
                h_inter = min(h_donors_1, h_acceptors_1) 
                h_intra_2 = 0
            else:
                x_2 = self.get_node_features(smiles_2)
                edge_attr_2, edge_index_2 = self.get_edge_features(smiles_2)  
                h_donors_2, h_acceptors_2 = self.get_hydrogen_bond(smiles_2)
                h_inter = min(h_donors_1, h_acceptors_1) + min(h_donors_2, h_acceptors_2)
                h_intra_2 = min(h_donors_2, h_acceptors_2)


            h_intra_1 = min(h_donors_1, h_acceptors_1)
            
            target  = []
            target.append([log_CMC])


            # Create PyTorch Geometric Data object
            
            data_1 = Data(x=x_1, 
                        edge_index=edge_index_1,
                        edge_attr=edge_attr_1,
                        T = T_norm, 
                        smiles_id_1 = smiles_1,
                        ratio_1 = ratio_1,
                        h_inter = h_inter,
                        h_intra_1 = h_intra_1,
                        y=torch.tensor(target, dtype=torch.float))
            
            data_2 = Data(x=x_2, 
                        edge_index=edge_index_2,
                        edge_attr=edge_attr_2,
                        T = T_norm, 
                        smiles_id_2 = smiles_2,
                        ratio_2 = ratio_2,
                        h_inter = h_inter,
                        h_intra_2 = h_intra_2,
                        y=torch.tensor(target, dtype=torch.float))
            

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list_1.append(data_1)
            data_list_2.append(data_2)

        torch.save(self.collate(data_list_1), self.processed_paths[0])
        torch.save(self.collate(data_list_2), self.processed_paths[1])
        
    
    

