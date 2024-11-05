# -*- coding: utf-8 -*-
"""
@author: BrozosCh
"""
from mg_smiles_to_molecules_ternary import MyOwnDataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch.nn import Sequential, Linear
from torch_scatter import scatter_add
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd 
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import Data, Batch
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape

# hyperparameters of the model
parser = argparse.ArgumentParser()
parser.add_argument('--plot', default = True) # Plot of train, validation and test error plots
parser.add_argument('--epochs', default=200)   # number of epochs
parser.add_argument('--dim', default=128)   # size of hidden node states
parser.add_argument('--lrate', default=0.001)   #  learning rate
parser.add_argument('--batch', default = 32)  # batch size
parser.add_argument('--split_type', default = 1) # Type of split described in the manuscript
parser.add_argument('--surfactant', default  = 'mega_10') # Surfactant to be excluded in split type 2
parser.add_argument('--early_stopping_patience', default=30)   # number of epochs until early stopping
parser.add_argument('--lrfactor', default=0.8)   # decreasing factor for learning rate
parser.add_argument('--lrpatience', default=3)   # number of consecutive epochs without model improvement after which learning rate is decreased


args, unknow = parser.parse_known_args()
plot = args.plot
epochs = int(args.epochs)
dim = int(args.dim)
lrate = float(args.lrate)
batch = int(args.batch)
lrfactor = float(args.lrfactor)
lrpatience = int(args.lrpatience)
early_stopping_patience = int(args.early_stopping_patience)
split_type = int(args.split_type)
surfactant = (args.surfactant)

# The GNN model architecture.

class GNNReg(torch.nn.Module):
    def __init__(self):
        super(GNNReg, self).__init__() 
        self.lin0 = Linear(dataset_molecule_1.num_features, dim)         # Initial linear transformation layer for node features
            
        self.transformation_layer = nn.Linear(dataset_molecule_1.num_edge_features, dim)  # Initial linear transformation layer for edge features
        gine_nn = Sequential(Linear(dim, int(dim*2)), nn.ReLU(), Linear(int(dim*2), dim))   
            
        self.conv1 = GINEConv(gine_nn, train_eps = False)  # The graph convolutinal layer   
        
        self.global_transformation_layer = nn.Linear(1, dim)
        global_gine_nn = Sequential(Linear(dim, int(dim*2)), nn.ReLU(), Linear(int(dim*2), dim))   
        self.global_conv_layer = GINEConv(global_gine_nn)

        self.fc1 = torch.nn.Linear(dim, dim)     # Initial layer of the MLP. The input dimension is increased by 1 neuron, to incorporate the temperature information.  
        self.fc2 = torch.nn.Linear(dim+1, dim)
        self.fc3 = torch.nn.Linear(dim , 1)

    def construct_graph(self, fp_1, fp_2, fp_3, h_inter, h_intra_1, ratio_2, ratio_3, h_inter_1_3, h_inter_2_3):
        data_list = []
        for i in range(fp_1.shape[0]):
            fp_1_obj = fp_1[i].view(1,-1)
            fp_2_obj = fp_2[i].view(1,-1)
            fp_3_obj = fp_3[i].view(1,-1)
            h_inter_obj = h_inter[i]
            h_intra_1_obj = h_intra_1[i]
            ratio_2_obj = ratio_2[i]
            ratio_3_obj = ratio_3[i]
            h_inter_1_3_obj = h_inter_1_3[i]
            h_inter_2_3_obj = h_inter_2_3[i]


            if (ratio_2_obj == 0) and (ratio_3_obj == 0):
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) 
                edge_attr = torch.tensor([[h_intra_1_obj], [h_intra_1_obj]], dtype= torch.float)
                x_obj = fp_1_obj.repeat(2,1)
                global_graph = Data(x=x_obj, edge_index=edge_index, edge_attr=edge_attr)
            elif (ratio_2_obj != 0) and (ratio_3_obj == 0):
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) 
                edge_attr = torch.tensor([[h_inter_obj], [h_inter_obj]], dtype= torch.float)
                x_obj = torch.cat((fp_1_obj, fp_2_obj), dim=0)
                global_graph = Data(x=x_obj,  edge_index=edge_index, edge_attr=edge_attr)
            else:
                edge_index = torch.tensor([[0, 1, 1 ,2 ,2 , 0], [1, 0,2,1,0,2]], dtype=torch.long) 
                edge_attr = torch.tensor([[h_inter_obj, h_inter_2_3_obj, h_inter_1_3_obj], [h_inter_obj, h_inter_2_3_obj,h_inter_1_3_obj]], dtype= torch.float)
                x_obj = torch.cat((fp_1_obj, fp_2_obj,fp_3_obj), dim=0)
                global_graph = Data(x=x_obj,  edge_index=edge_index, edge_attr=edge_attr)

            
            data_list.append(global_graph)

        batch_obj = Batch.from_data_list(data_list)
        return batch_obj


        
    
    def forward(self, data_molecule_1, data_molecule_2, data_molecule_3):
        x_1, edge_index_1, edge_attr_1, temp, ratio_1  = data_molecule_1.x , data_molecule_1.edge_index, data_molecule_1.edge_attr, data_molecule_1.T, data_molecule_1.ratio_1
        x_2, edge_index_2, edge_attr_2, temp_2, ratio_2  = data_molecule_2.x , data_molecule_2.edge_index, data_molecule_2.edge_attr, data_molecule_2.T, data_molecule_2.ratio_2
        x_3, edge_index_3, edge_attr_3, temp_2, ratio_3  = data_molecule_3.x , data_molecule_3.edge_index, data_molecule_3.edge_attr, data_molecule_3.T, data_molecule_3.ratio_3
        h_inter, h_intra_1 = data_molecule_1.h_inter, data_molecule_1.h_intra_1
        h_inter_1_3, h_inter_2_3 =  data_molecule_3.h_inter_1_3 , data_molecule_3.h_inter_2_3


      # molecule 1
        x_1 = F.relu(self.lin0(x_1))
        x_1 = F.relu(self.conv1(x_1, edge_index_1, edge_attr = self.transformation_layer(edge_attr_1)))
        x_1 = scatter_add(x_1, data_molecule_1.batch, dim=0)
        x_1 = x_1 * ratio_1.unsqueeze(1)


      # molecule 2
        x_2 = F.relu(self.lin0(x_2))
        x_2 = F.relu(self.conv1(x_2, edge_index_2, edge_attr = self.transformation_layer(edge_attr_2)))
        x_2 = scatter_add(x_2, data_molecule_2.batch, dim=0)
        x_2 = x_2 * ratio_2.unsqueeze(1)

        x_3 = F.relu(self.lin0(x_3))
        x_3 = F.relu(self.conv1(x_3, edge_index_3, edge_attr = self.transformation_layer(edge_attr_3)))
        x_3 = scatter_add(x_3, data_molecule_3.batch, dim=0)
        x_3 = x_3 * ratio_3.unsqueeze(1)


        global_graph = self.construct_graph(fp_1= x_1, fp_2= x_2, fp_3 = x_3, h_inter= h_inter,
                                             h_intra_1= h_intra_1, ratio_2 = ratio_2, ratio_3= ratio_3, h_inter_1_3 = h_inter_1_3, h_inter_2_3 = h_inter_2_3)


        x_glob = F.relu(self.global_conv_layer(global_graph.x, global_graph.edge_index,
                                               edge_attr= self.global_transformation_layer(global_graph.edge_attr)))

        fp = scatter_add(x_glob, global_graph.batch, dim=0)

        temp = 10*temp  # The temperature was originally normalized between {0,1}. As described in our work, we found the normalization between {0,10} to perform better. Therefore, this extra step was added.

        x = F.relu(self.fc1(fp))
        x = torch.cat([x, temp.reshape(x.shape[0],1)], dim = 1) 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The minimum and maximum temperatures are neccessary values for temperature unscaling. The oroginal value is calculated as:  temp = scaled_value * (max_temp - min_temp) + min_temp`

min_temp = torch.tensor(273.15, dtype = torch.float32)
max_temp = torch.tensor(363.15, dtype = torch.float32) 


# Load the training data set of comp-inter to get the mean and std values and the external ternary test data set. No model re-training is performed.

if split_type == 1:
    dataset_molecule_1 = MyOwnDataset(root = r'split_type_1\Train', molecule = 1) # Load the training data set of comp-inter to get the mean and std values. 
    ext_test_dataset_molecule_1 = MyOwnDataset(root = r'Test', molecule = 1) 
    ext_test_dataset_molecule_2 = MyOwnDataset(root = r'Test', molecule = 2)
    ext_test_dataset_molecule_3 = MyOwnDataset(root = r'Test', molecule = 3)
else:
    print('Error in split type')


dataset_molecule_1.data.y = dataset_molecule_1.data.y
ext_test_dataset_molecule_1.data.y = ext_test_dataset_molecule_1.data.y

# Normalization of the target property

mean = torch.as_tensor(dataset_molecule_1.data.y, dtype=torch.float).mean()
std = torch.as_tensor(dataset_molecule_1.data.y, dtype=torch.float).std()
dataset_molecule_1.data.y = (dataset_molecule_1.data.y - mean) / std
ext_test_dataset_molecule_1.data.y = (ext_test_dataset_molecule_1.data.y - mean) / std



def data_preparation(seed):
    torch.manual_seed(seed)
    test_loader_1 = DataLoader(ext_test_dataset_molecule_1[:], batch_size= len(ext_test_dataset_molecule_1))
    test_loader_2 = DataLoader(ext_test_dataset_molecule_2[:], batch_size= len(ext_test_dataset_molecule_1))
    test_loader_3 = DataLoader(ext_test_dataset_molecule_3[:], batch_size= len(ext_test_dataset_molecule_1))
    return test_loader_1, test_loader_2, test_loader_3

def ensemble(loader_1, loader_2, loader_3, model, ensemble_path, seed):
    model = GNNReg().to(device)
    model.load_state_dict(torch.load(ensemble_path+'base_model_{}.pt'.format(seed), map_location= torch.device('cpu')))
    model.eval()
    for data_molecule_1, data_molecule_2, data_molecule_3 in zip(loader_1, loader_2,loader_3):
        data_molecule_1 = data_molecule_1.to(device)
        data_molecule_2 = data_molecule_2.to(device)
        data_molecule_3 = data_molecule_3.to(device)
        out = model(data_molecule_1, data_molecule_2,data_molecule_3)
        out_unstandardized = out*std + mean
    return out_unstandardized

# User defined path. The path should lead to the 20 trained models from the comp-inter split, as described in the manuscript.
ensemble_path = ("results_{}\\".format(split_type))


def ensemble_calculations(ensemble_path = ensemble_path):
    y_pred = []
    for counter in range(1,21):
        test_loader_1, test_loader_2,test_loader_3 = data_preparation(seed=counter)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GNNReg().to(device)
        model.load_state_dict(torch.load(ensemble_path+'base_model_{}.pt'.format(counter), map_location= torch.device('cpu')))
        model.eval()
        y_pred.append(ensemble(test_loader_1, test_loader_2,test_loader_3, model, ensemble_path, seed = counter))
    concatenated_tensor = torch.cat(y_pred, dim = 1)
    row_means = torch.mean(concatenated_tensor, dim = 1)
    y_mean = row_means.detach().numpy()
    y_mean = [round(x, 3) for x in y_mean]
    y_true = np.array(ext_test_dataset_molecule_1.data.y.cpu())
    y_true = y_true*std.item() + mean.item()
    y_true = [round(x, 3) for x in y_true.flatten()]
    mae = round(mean_absolute_error(y_true, y_mean),5)
    rmse = round(np.sqrt(mean_squared_error(y_true, y_mean)),5)
    mape_out = 100*round(mape(y_true, y_mean),5)
    r_2 = r2_score(y_true, y_mean)
    print("State of the art ensemble MAE:{},  State of the art ensemble RMSE:{}, State of the art ensemble MAPE:{}".format(mae,rmse,mape))
    return mae,rmse,mape_out, y_true, y_mean, r_2
