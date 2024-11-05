# -*- coding: utf-8 -*-
"""
@author: BrozosCh
"""
from mg_smiles_to_molecules import MyOwnDataset
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
from utils import EarlyStopping, error_metrics
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
parser.add_argument('--split_type', default = 1) # Type of split described in the manuscript5
parser.add_argument('--surfactant', default  = 'cetylpyridinium_chloride') # Surfactant to be excluded in split type 3 (sub-test set), referred to as mix-surf-extra in the manuscript
parser.add_argument('--sub_set', default  = 1) # Subset to be consider in the second split type, referred to as mix-comp-extra in the manuscript
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
sub_set = int(args.sub_set)

# The GNN model architecture.

class GNNReg(torch.nn.Module):
    def __init__(self):
        super(GNNReg, self).__init__() 
        self.lin0 = Linear(dataset_molecule_1.num_features, dim)         # Initial linear transformation layer for node features
            
        self.transformation_layer = nn.Linear(dataset_molecule_1.num_edge_features, dim)  # Initial linear transformation layer for edge features
        gine_nn = Sequential(Linear(dim, int(dim*2)), nn.ReLU(), Linear(int(dim*2), dim))   
            
        self.conv1 = GINEConv(gine_nn, train_eps = False)  # The graph convolutinal layer   
        
        self.mixture_transformation_layer = nn.Linear(1, dim)
        mixture_gine_nn = Sequential(Linear(dim, int(dim*2)), nn.ReLU(), Linear(int(dim*2), dim))   
        self.mixture_conv_layer = GINEConv(mixture_gine_nn)

        self.fc1 = torch.nn.Linear(dim, dim)      
        self.fc2 = torch.nn.Linear(dim+1, dim)
        self.fc3 = torch.nn.Linear(dim , 1)

    def construct_graph(self, fp_1, fp_2, h_inter, h_intra_1, ratio_2):
        data_list = []
        for i in range(fp_1.shape[0]):               #Iterate through each object of the batch
            fp_1_obj = fp_1[i].view(1,-1)
            fp_2_obj = fp_2[i].view(1,-1)
            h_inter_obj = h_inter[i]
            h_intra_1_obj = h_intra_1[i]
            ratio_2_obj = ratio_2[i]

            if ratio_2_obj == 0 :                # In the case of single surfactants, x2 = 0, a mixture graph with two nodes is constructed where the edge-features account for the intra-molecular interactions.
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) 
                edge_attr = torch.tensor([[h_intra_1_obj], [h_intra_1_obj]], dtype= torch.float)
                x_obj = fp_1_obj.repeat(2,1)
                global_graph = Data(x=x_obj, edge_index=edge_index, edge_attr=edge_attr)

            else:                   # In the case of two surfactants, x2 ≠ 0, a mixture graph with two nodes is constructed where the edge-features account for the intermolecular interactions.
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) 
                edge_attr = torch.tensor([[h_inter_obj], [h_inter_obj]], dtype= torch.float)
                x_obj = torch.cat((fp_1_obj, fp_2_obj), dim=0)
                global_graph = Data(x=x_obj,  edge_index=edge_index, edge_attr=edge_attr)
            
            data_list.append(global_graph)

        batch_obj = Batch.from_data_list(data_list)
        return batch_obj


        
    
    def forward(self, data_molecule_1, data_molecule_2):
        x_1, edge_index_1, edge_attr_1, temp, ratio_1  = data_molecule_1.x , data_molecule_1.edge_index, data_molecule_1.edge_attr, data_molecule_1.T, data_molecule_1.ratio_1
        x_2, edge_index_2, edge_attr_2, temp_2, ratio_2  = data_molecule_2.x , data_molecule_2.edge_index, data_molecule_2.edge_attr, data_molecule_2.T, data_molecule_2.ratio_2
        h_inter, h_intra_1 = data_molecule_1.h_inter, data_molecule_1.h_intra_1


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

      # Construction of a global graph 
        mixture_graph = self.construct_graph(fp_1= x_1, fp_2= x_2, h_inter= h_inter,
                                             h_intra_1= h_intra_1, ratio_2 = ratio_2)


        x_glob = F.relu(self.mixture_conv_layer(mixture_graph.x, mixture_graph.edge_index,
                                               edge_attr= self.mixture_transformation_layer(mixture_graph.edge_attr)))

        fp = scatter_add(x_glob, mixture_graph.batch, dim=0)

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


# Load the task's corresponding dataset

if split_type == 1:
    dataset_molecule_1 = MyOwnDataset(root = r'split_type_1\Train', molecule = 1)
    dataset_molecule_2 = MyOwnDataset(root = r'split_type_1\Train', molecule = 2)
    ext_test_dataset_molecule_1 = MyOwnDataset(root = r'split_type_1\Test', molecule = 1)
    ext_test_dataset_molecule_2 = MyOwnDataset(root = r'split_type_1\Test', molecule = 2)
elif split_type == 2:
    dataset_molecule_1 = MyOwnDataset(root = r'split_type_2\subset_{}\Train'.format(sub_set), molecule = 1)
    dataset_molecule_2 = MyOwnDataset(root = r'split_type_2\subset_{}\Train'.format(sub_set), molecule = 2)
    ext_test_dataset_molecule_1 = MyOwnDataset(root = r'split_type_2\subset_{}\Test'.format(sub_set), molecule = 1)
    ext_test_dataset_molecule_2 = MyOwnDataset(root = r'split_type_2\subset_{}\Test'.format(sub_set), molecule = 2)
elif split_type == 3:
    dataset_molecule_1 = MyOwnDataset(root = r'split_type_3\{}\Train'.format(surfactant), molecule = 1)
    dataset_molecule_2 = MyOwnDataset(root = r'split_type_3\{}\Train'.format(surfactant), molecule = 2)
    ext_test_dataset_molecule_1 = MyOwnDataset(root = r'split_type_3\{}\Test'.format(surfactant), molecule = 1)
    ext_test_dataset_molecule_2 = MyOwnDataset(root = r'split_type_3\{}\Test'.format(surfactant), molecule = 2)
elif split_type == 4:
    dataset_molecule_1 = MyOwnDataset(root = r'split_type_4\Train', molecule = 1)
    dataset_molecule_2 = MyOwnDataset(root = r'split_type_4\Train', molecule = 2)
    ext_test_dataset_molecule_1 = MyOwnDataset(root = r'split_type_4\Test', molecule = 1)
    ext_test_dataset_molecule_2 = MyOwnDataset(root = r'split_type_4\Test', molecule = 2)
else:
    print('The inputed split type does not exist.')


dataset_molecule_1.data.y = dataset_molecule_1.data.y
ext_test_dataset_molecule_1.data.y = ext_test_dataset_molecule_1.data.y

# Normalization of the target property

mean = torch.as_tensor(dataset_molecule_1.data.y, dtype=torch.float).mean()
std = torch.as_tensor(dataset_molecule_1.data.y, dtype=torch.float).std()
dataset_molecule_1.data.y = (dataset_molecule_1.data.y - mean) / std
ext_test_dataset_molecule_1.data.y = (ext_test_dataset_molecule_1.data.y - mean) / std



def data_preparation(seed, dataset_molecule_1 = dataset_molecule_1, dataset_molecule_2 = dataset_molecule_2):
    torch.manual_seed(seed)
    dataset_molecule_1, perm = dataset_molecule_1.shuffle(return_perm= True)
    dataset_molecule_2 = dataset_molecule_2[perm]
    val_dataset_1, val_dataset_2 = dataset_molecule_1[:385], dataset_molecule_2[:385]
    train_dataset_1, train_dataset_2 = dataset_molecule_1[385:], dataset_molecule_2[385:]
    train_loader_1, train_loader_2 = DataLoader(train_dataset_1, batch_size=batch), DataLoader(train_dataset_2, batch_size=batch)
    val_loader_1, val_loader_2 = DataLoader(val_dataset_1, batch_size=batch), DataLoader(val_dataset_2, batch_size=batch)
    test_loader_1 = DataLoader(ext_test_dataset_molecule_1[:], batch_size= len(ext_test_dataset_molecule_1))
    test_loader_2 = DataLoader(ext_test_dataset_molecule_2[:], batch_size= len(ext_test_dataset_molecule_1))
    return train_loader_1, train_loader_2, val_loader_1, val_loader_2, test_loader_1, test_loader_2


def train(loader_1, loader_2, model, optimizer):
    model.train()
    loss_all = abs_loss_all = total_examples = 0
    norm_train_mae, train_mae = 0, 0
    temp_train_predictions,temp_train_real = [], []
    
    for data_molecule_1, data_molecule_2 in zip(loader_1, loader_2):
        data_molecule_1 = data_molecule_1.to(device)
        data_molecule_2 = data_molecule_2.to(device)

        optimizer.zero_grad()
        out = model(data_molecule_1, data_molecule_2)
        target = data_molecule_1.y.view(-1,1)

        loss = F.mse_loss(out, target)
        loss.backward()
        loss_all += loss * data_molecule_1.num_graphs
        total_examples += data_molecule_1.num_graphs
        optimizer.step()
        
        #Standardized errors calculation. They are calculated but not returned.
        norm_train_rmse = torch.sqrt(loss_all/total_examples)
        norm_train_mae += (out - target).abs().sum(0).item()  
        
        #calculating the unstandardized errors
        out_standardized = out*std + mean
        target_standardized = target*std + mean

        abs_loss = F.mse_loss(out_standardized, target_standardized)  # The MSE loss on the unstandardized data is calculated
        abs_loss_all += abs_loss*data_molecule_1.num_graphs

        temp_train_predictions.append(out_standardized)
        temp_train_real.append(target_standardized)
        
        train_rmse = torch.sqrt(abs_loss_all/total_examples)  
        train_mae += (out_standardized - target_standardized).abs().sum(0).item()
    
    predicted_train = torch.cat(temp_train_predictions, dim = 0)
    real_train = torch.cat(temp_train_real, dim = 0)
    

     #We report only the undstandardized errors   
    return abs_loss_all / total_examples,  train_rmse.item(), train_mae / total_examples

def test(loader_1, loader_2, model):
    model.eval()
    val_mae, val_rmse = 0, 0
    loss_all_norm = norm_val_mae = abs_loss_all = 0 
    loss_all = total_examples = 0
    temp_test_predictions, temp_test_real = [],[]
    
    with torch.no_grad():
        for data_molecule_1, data_molecule_2 in zip(loader_1, loader_2):
            data_molecule_1 = data_molecule_1.to(device)
            data_molecule_2 = data_molecule_2.to(device)

            out = model(data_molecule_1, data_molecule_2)
            target = data_molecule_1.y.view(-1,1)
        
            loss = F.mse_loss(out, target)
            loss_all += loss * data_molecule_1.num_graphs
            total_examples += data_molecule_1.num_graphs
       
            #calculating standardized errors
            norm_val_rmse = torch.sqrt(loss_all/total_examples)
            norm_val_mae += (out - target).abs().sum(0).item()
            
            #calculating the unstandardized errors
            out_standardized = out*std + mean
            target_data_standardized = target*std + mean

            abs_loss = F.mse_loss(out_standardized, target_data_standardized)
            abs_loss_all += abs_loss*data_molecule_1.num_graphs

            val_rmse = torch.sqrt(abs_loss_all/total_examples) 
            val_mae += (out_standardized - target_data_standardized).abs().sum(0).item()
            
            temp_test_predictions.append(out_standardized)
            temp_test_real.append(target_data_standardized)
            
            
    predicted_test = torch.cat(temp_test_predictions, dim = 0)
    real_test = torch.cat(temp_test_real, dim = 0)
    
    _, _, test_mape = error_metrics(real_test.detach().numpy(), predicted_test.detach().numpy())
    #We report only the unstandardized errors
    return abs_loss_all / total_examples , val_rmse.item(), val_mae / total_examples, test_mape*100

# Write predictions on the given dataset in an Excel file, together with the corresponding Smiles string. The results are been printed in an Excel File.

def write_predictions(loader_1, loader_2, model, save_path, dataset_type,counter):
    model = model
    model.load_state_dict(torch.load(save_path+'base_model_{}.pt'.format(counter)))
    model.eval()
    smiles_1, smiles_2, ratio_1, ratio_2, predicted, measured, temperature = [],[],[], [], [],[],[]
    df_exp = pd.DataFrame()
    pred, real_value, pred_list, rel_error = None, None,  [], []
    for data_1, data_2 in zip(loader_1, loader_2):
        for mol_1, mol_2 in zip(data_1.smiles_id_1, data_2.smiles_id_2):
            smiles_1.append(mol_1)
            smiles_2.append(mol_2)
        for rat_1, rat_2 in zip(data_1.ratio_1, data_2.ratio_2):
            ratio_1.append(rat_1.item())
            ratio_2.append(rat_2.item())
        data_1 = data_1.to(device)
        data_2 = data_2.to(device)
        real_value = data_1.y*std + mean
        real_value = real_value.view(-1,1)

        pred = model(data_1, data_2)
        pred_norm = pred*std + mean

        real_diff = abs( (pred_norm - real_value) / real_value)
        normalized_temp = data_1.T *(max_temp - min_temp) + min_temp
        
        for k in pred_list:
            smiles_1.append(k[0])
            smiles_2.append(k[1])
            ratio_1.append(round(k[2],3))
            ratio_2.append(round(k[3],3))
            predicted.append(round(k[4],3))
            measured.append(round(k[5],3))
        for c in real_value:
            measured.append(round(c.item(),3))
        for k in pred_norm:
            predicted.append(round(k.item(),3))
        
        for d in real_diff:
            rel_error.append(d.item())

        for t in normalized_temp:
            temperature.append(round(t.item(),2))
    
    y_abs_meas = [round((10**x)/1000,3) for x in measured]
    y_abs_pred = [round((10**x)/1000,3) for x in predicted]
            
    df_exp['SMILES_1'] = smiles_1
    df_exp['SMILES_2'] = smiles_2
    df_exp['Ratio_1'] = ratio_1
    df_exp['Ratio_2'] = ratio_2
    df_exp['Predicted'] = predicted
    df_exp['Measured'] = measured
    df_exp['Predicted_unscaled'] = y_abs_pred
    df_exp['Measured_unscaled'] = y_abs_meas
    df_exp['Temperature'] = temperature
    df_exp.to_excel(str(save_path)+str(dataset_type)+'_base_model_{}.xlsx'.format(counter), index = False)

#User defined path.
     
save_path = str('results_{}\\'.format(split_type))

    
# Function for saving the best model's parameters.

def save_checkpoint(model,filename):
    print('Saving checkpoint') 
    torch.save(model.state_dict(), save_path+filename)    
    time.sleep(1.5)

# This function returns the unstandardized predictions. Based on the seed, we can initiate it with different models.

def ensemble(loader_1, loader_2, model, ensemble_path, seed):
    model = GNNReg().to(device)
    model.load_state_dict(torch.load(ensemble_path+'base_model_{}.pt'.format(seed), map_location= torch.device('cpu')))
    model.eval()
    for data_molecule_1, data_molecule_2 in zip(loader_1, loader_2):
        data_molecule_1 = data_molecule_1.to(device)
        data_molecule_2 = data_molecule_2.to(device)
        out = model(data_molecule_1, data_molecule_2)
        out_unstandardized = out*std + mean
    return out_unstandardized

#User defined path, where the trained models are saved.

ensemble_path = ('results_{}\\'.format(split_type))

def ensemble_calculations(ensemble_path = ensemble_path):
    y_pred = []
    for counter in range(1,21):
        _,_,_,_,test_loader_1, test_loader_2 = data_preparation(seed=counter)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GNNReg().to(device)
        model.load_state_dict(torch.load(ensemble_path+'base_model_{}.pt'.format(counter), map_location= torch.device('cpu')))
        model.eval()
        y_pred.append(ensemble(test_loader_1, test_loader_2, model, ensemble_path, seed = counter))
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

# This function create different training-validation splits, initiates the model training, saves the model in the best epoch and can also save the predictions on the test set for ensemble predictions.
def training(counter):
    best_epoch, best_val_rmse , best_epoch_test_rmse = None, None, None
    best_val_mae, best_epoch_test_mae = None, None
    best_val_mape, best_epoch_test_mape = None, None
    
    train_losses, val_errors, test_errors = [], [],[]
    train_errors, val_losses, test_losses = [],[], []
    
    train_loader_1, train_loader_2, validation_loader_1, validation_loader_2, test_loader_1, test_loader_2 = data_preparation(seed=counter)
    model = GNNReg().to(device)
    print(model)
    
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = lrate)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lrfactor, patience=lrpatience, min_lr=0.0000001)
     

    for epoch in range(1,epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss, train_rmse, train_mae = train(train_loader_1, train_loader_2, model, optimizer)
        val_loss, val_rmse, val_mae, val_mape = test(validation_loader_1, validation_loader_2, model)
        test_loss, test_rmse, test_mae, test_mape = test(test_loader_1, test_loader_2, model)    

        print('Epoch: {} , Learning Rate: {}, Train error: {:.4f}, Val. error: {:.4f}, Test error: {:.4f}'.format(epoch, lr, train_loss, val_loss, test_loss))
        print('Val: val_rmse {:.4f}, val_mae {:.4f}, Test: test_rmse {:.4f}, test_mae {:.4f}'.format( val_rmse, val_mae, test_rmse, test_mae))
        

        scheduler.step(val_loss)
        train_losses.append(train_loss.detach().numpy())
        val_losses.append(val_loss.detach().numpy())
        test_losses.append(test_loss.detach().numpy())
        
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)
        test_errors.append(test_rmse)        
        
        if epoch > 0:
            if  best_val_rmse is None:
                best_epoch = epoch
                best_val_rmse, best_epoch_test_rmse = val_rmse, test_rmse
                best_val_mae, best_epoch_test_mae = val_mae, test_mae
                best_val_mape, best_epoch_test_mape =  val_mape, test_mape
            elif val_rmse < best_val_rmse:
                best_epoch = epoch
                best_val_rmse, best_epoch_test_rmse = val_rmse, test_rmse
                best_val_mae, best_epoch_test_mae = val_mae, test_mae
                best_val_mape, best_epoch_test_mape =  val_mape, test_mape

                save_checkpoint(model,'base_model_{}.pt'.format(counter))

        
            early_stopping(val_rmse)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    print('Best model with respect to validation error in epoch {:03d} with \nVal RMSE {:.5f}\nTest RMSE {:.5f}\n'.format(best_epoch, best_val_rmse, best_epoch_test_rmse))  
    
   # write_predictions(validation_loader_1, validation_loader_2, model, save_path, 'val', counter = counter)
   # write_predictions(test_loader_1, test_loader_2,model, save_path, 'test', counter = counter)
 
   
    if plot is True:
        plt.title('Total loss')
        plt.plot(range(1,len(train_losses)+1), train_losses, label = 'Train')
        plt.plot(range(1,len(val_losses)+1), val_losses, label = 'Validation')
        plt.plot(range(1,len(test_losses)+1), test_losses, label='Test')
        plt.ylim(0,1.5)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()
               
        plt.title('RMSE')
        plt.plot(range(1,len(train_errors)+1), train_errors, label = 'Train')
        plt.plot(range(1,len(val_errors)+1), val_errors, label = 'Validation')
        plt.plot(range(1,len(test_errors)+1), test_errors, label='Test')
        plt.ylim(0,1.5)
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()
        
    else:
        pass
    return train_loss, val_rmse, val_mae, test_rmse, test_mae, best_val_rmse, best_epoch_test_rmse, best_val_mae, best_epoch_test_mae, val_mape, test_mape, best_val_mape, best_epoch_test_mape, val_loss, test_loss

# In this section, we define the number of runs we wish (40 during our work) and append the parameters to corresponding lists. Each run has a different training validation split.
val_rmse_40, test_rmse_40  = [], []
val_mae_40, test_mae_40 = [], []
val_mape_40, test_mape_40 = [], []
best_val_rmse_40, best_epoch_test_rmse_40 = [], []
best_val_mae_40, best_epoch_test_mae_40 = [], []
best_val_mape_40, best_epoch_test_mape_40 = [], []


def control_fun():
    for i in range(1, 21):
        out = training(i)
        test_rmse_40.append(out[3]) 
        test_mae_40.append(out[4])
        val_rmse_40.append(out[1])
        val_mae_40.append(out[2])
        best_val_rmse_40.append(out[5])
        best_epoch_test_rmse_40.append(out[6])
        best_val_mae_40.append(out[7])
        best_epoch_test_mae_40.append(out[8])
        val_mape_40.append(out[9])
        test_mape_40.append(out[10])
        best_val_mape_40.append(out[11])
        best_epoch_test_mape_40.append(out[12])



control_fun()

#Optionally the results are saved in a dataframe. 

df = pd.DataFrame()
df['val_rmse_40'] = val_rmse_40
df['val_mae_40'] =  val_mae_40
df['val_mape_40'] = val_mape_40
df['test_rmse_40'] = test_rmse_40
df['test_mae_40'] = test_mae_40
df['test_mape_40'] = test_mape_40
df['best_val_rmse_40'] = best_val_rmse_40
df['best_epoch_test_rmse_40'] = best_epoch_test_rmse_40
df['best_val_mae_40'] = best_val_mae_40
df['best_epoch_test_mae_40'] = best_epoch_test_mae_40
df['best_val_mape_40'] = best_val_mape_40
df['best_epoch_test_mape_40'] = best_epoch_test_mape_40
df['Learning_rate'] = lrate
df['Batch_size'] = batch
df.loc['mean'] = df.mean()

