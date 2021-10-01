import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn import metrics

import glob

class Net_fc(nn.Module):
    
    known_ou_activation = {"none":lambda x : x, \
                           "sigmoid":nn.Sigmoid(),\
                           "softmax":nn.Softmax(dim=1),\
                           "relu":nn.ReLU(),\
                           "sin":torch.sin,\
                          }
    
    def __init__(self, indim, layer_sizes, ou_activation="none", dropout_ratio=None, verbos=False):
        super().__init__()
        
        self.verbos = verbos
        self.indim = indim
        self.layer_sizes = layer_sizes
        
        if(ou_activation not in self.known_ou_activation):
            raise Exception("Unknow output activation function")
        self.ou_activation_fun = self.known_ou_activation[ou_activation]
        
        if(dropout_ratio!=None):
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        
        if(verbos):
            print("Input dim = ", indim)
            print("Layer sizes = ", layer_sizes)
            print("Output activation = ", self.ou_activation_fun)
            print("Dropout = ", self.dropout)
            
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)):
            if(i==0):
                layer_i = nn.Linear(indim, layer_sizes[i])
            else:
                layer_i = nn.Linear(layer_sizes[i-1], layer_sizes[i])
            self.layers.append(layer_i)
            if(verbos):
                print("Layer %d"%(i), layer_i)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.layers)):
            layer_i = self.layers[i]
            x = layer_i(x)
            if(i!=len(self.layers)-1):
                x = F.relu(x)
                if(self.dropout!=None):
                    x = self.dropout(x)
        return self.ou_activation_fun(x)


class FlServer:
    
    known_loss_fun = {"mse":nn.MSELoss(), "cross_entropy":nn.CrossEntropyLoss()}
    
    def __init__(self, indim, oudim, nz, layer_sizes, ou_activation="none", dropout_ratio=None, loss="mse", verbos=False):
        self.model = Net_fc(indim, layer_sizes+[oudim], ou_activation=ou_activation, dropout_ratio=dropout_ratio, verbos=verbos)
        self.param_keys = self.model.state_dict().keys()
        self.loss = self.known_loss_fun[loss]
                
        ## for fine tune
        self.nz = nz
        self.models_ft = []
        for j in range(self.nz):
            model_j = Net_fc(indim, layer_sizes+[oudim], ou_activation=ou_activation, dropout_ratio=dropout_ratio, verbos=verbos)
            self.models_ft.append(model_j)
    
        
    def get_central_param(self):
        return self.model.state_dict()
    
    def get_central_params_ft(self):
        return [model.state_dict() for model in self.models_ft]
    
    def set_central_param(self, param):
        self.model.load_state_dict(param)
    
    def set_central_params_ft(self, params):
        for j in range(self.nz):
            self.models_ft[j].load_state_dict(params[j])
    
    def update_central(self, param_list, weight_list):
        param_next = self.fed_avg(param_list, weight_list)
#         self.model.load_state_dict(param_next)
        self.set_central_param(param_next)
    
    def update_central_ft(self, params_list, weights_list):
        params_next = []
        for j in range(self.nz):
            param_list_j = [params[j] for params in params_list]
            weight_list_j = [weights[j] for weights in weights_list]
            
            param_next_j = self.fed_avg(param_list_j, weight_list_j)
#             self.models_ft[j].load_state_dict(param_next_j)
            params_next.append(param_next_j)
        self.set_central_params_ft(params_next)
        
    def fed_avg(self, param_list, weight_list):
        param_avg = {}
        weight_tot = np.sum(weight_list)
        
        #print("weight_list", weight_list)
        
        for param, weight in zip(param_list, weight_list):
            assert param.keys() == self.param_keys
            for k in param:
                if not k in param_avg:
                    param_avg[k] = 1./weight_tot*weight*param[k]
                else:
                    param_avg[k] += 1./weight_tot*weight*param[k]
        
        return param_avg
    
    def run_model(self, model, loader):
        preds = []
        labels = []
        
        for i, data in enumerate(loader, 0):
            X, y = data
            y_pred = model.forward(X)
            
            preds = np.append(preds, y_pred.detach().numpy()[:,1])
            labels = np.append(labels, y.detach().numpy())
            
        return labels, preds
            

    def test(self, loaders):
        
        count_z = []
        acc_z = []
        log_loss_z = []
        
        labels_z = []
        preds_z = []
        
        labels = []
        preds = []
        for j in range(self.nz):
            labels_j, preds_j = self.run_model(self.model, loaders[j])
            count_z.append(len(labels_j))
            log_loss_z.append(metrics.log_loss(labels_j, preds_j))
            acc_z.append(metrics.accuracy_score(labels_j, 1*(preds_j>0.5)))
            
            labels_z.append(labels_j)
            preds_z.append(preds_j)
            
            labels = np.append(labels, labels_j)
            preds = np.append(preds, preds_j)

        res = {}
        res["log_loss"] = metrics.log_loss(labels, preds)
        res["acc"] = metrics.accuracy_score(labels, 1*(preds>0.5))
        res["count"] = len(labels)
        
        res["log_loss_z"] = log_loss_z
        res["acc_z"] = acc_z
        res["count_z"] = count_z
        
        res['labels_z'] = labels_z
        res['preds_z'] = preds_z
        
        return res
        
    
    def test_ft(self, loaders):
        
        count_z = []
        acc_z = []
        log_loss_z = []
    
        labels_z = []
        preds_z = []
        
        labels = []
        preds = []
        for j in range(self.nz):
            labels_j, preds_j = self.run_model(self.models_ft[j], loaders[j])
            
            count_z.append(len(labels_j))
            log_loss_z.append(metrics.log_loss(labels_j, preds_j))
            acc_z.append(metrics.accuracy_score(labels_j, 1*(preds_j>0.5)))

            labels_z.append(labels_j)
            preds_z.append(preds_j)
            
            labels = np.append(labels, labels_j)
            preds = np.append(preds, preds_j)

        res = {}
        res["log_loss"] = metrics.log_loss(labels, preds)
        res["acc"] = metrics.accuracy_score(labels, 1*(preds>0.5))
        res["count"] = len(labels)
        
        res["log_loss_z"] = log_loss_z
        res["acc_z"] = acc_z
        res["count_z"] = count_z
        
        res['labels_z'] = labels_z
        res['preds_z'] = preds_z
        
        return res
            

class FlClient:

    known_loss_fun = {"mse":nn.MSELoss(), "cross_entropy":nn.CrossEntropyLoss()}
    
    def __init__(self,indim, oudim, nz, layer_sizes, ou_activation="none", dropout_ratio=None, loss="mse", verbos=False):
        self.model = Net_fc(indim, layer_sizes+[oudim], ou_activation=ou_activation, dropout_ratio=dropout_ratio, verbos=verbos)
        
        self.loss_fun = self.known_loss_fun[loss]
        
        self.n_batch = 8
        
        self.weight = None
        self.weights_ft = None
        
        ## for fine tune
        self.nz = nz
        self.models_ft = []
        for j in range(self.nz):
            model_j = Net_fc(indim, layer_sizes+[oudim], ou_activation=ou_activation, dropout_ratio=dropout_ratio, verbos=verbos)
            self.models_ft.append(model_j)
        
    def get_local_param(self):
        return self.model.state_dict(), self.weight
    
    def get_local_params_ft(self):
        params = [model_j.state_dict() for model_j in self.models_ft]
        return params, self.weights_ft
    
    def set_local_param(self, param):
        self.model.load_state_dict(param)
    
    def set_local_params_ft(self, params):
        for j in range(self.nz):
            self.models_ft[j].load_state_dict(params[j])
    
    def train_local_model(self, model, train_loader, n_epoch):

        optimizer = optim.Adam(model.parameters())
        count = 0
        for i_epoch in range(n_epoch):
            for i, data in enumerate(train_loader, 0):
                X, y = data   
                y_pred = model.forward(X)
            

                loss_net = self.loss_fun(y_pred, y)
                loss_tot = loss_net

                loss_tot.backward()
                optimizer.step()

                if i_epoch == 0:
                    count += X.shape[0]
                    
        return {"count": count}
        
        
    def update_local(self, train_loader, n_epoch):
        res = self.train_local_model(self.model, train_loader, n_epoch)
        self.weight = res["count"]
        
    def update_local_ft(self, train_loaders, n_epoch):
        self.weights_ft = []
        for j in range(self.nz):
            loader_j = train_loaders[j]
            model_j = self.models_ft[j]
            res = self.train_local_model(model_j, loader_j, n_epoch)
            self.weights_ft.append(res["count"])
        

## synchrous fl pipeline

class SyncFl:
    
    def __init__(self, folder, x_dim, nz, c=1.0, n_batch=16):
        self.x_dim = x_dim
        self.nz = 2
        self.c = c
        self.n_batch = n_batch
        
        path_list = sorted(glob.glob(folder+"/*_*"))
        path_valid = folder + "/valid.npz"
        path_test = folder + "/test.npz"
        
        self.n_client = len(path_list)
        
        print(f"Found {self.n_client} data files:")
        for path in path_list:
            print(f"  {path}")
        print(f"Creating {self.n_client} clients.\n")
        
        ##---- 1. read in all the data, create loader for all of them
        self.loader_list = []
        for path_i in path_list:
            loader_i = self.make_loader(path_i)
            self.loader_list.append(loader_i)
        
        self.valid_loader = self.make_loader(path_valid)
        self.test_loader = self.make_loader(path_test)
        
        ##---- 2. create server and clients
        self.server = FlServer(self.x_dim, 2, 2, [64, 64], ou_activation="softmax", loss="cross_entropy", verbos=False)
        
        self.client_list = []
        for i in range(self.n_client):
            client_i = FlClient(self.x_dim, 2, 2, [64, 64], ou_activation="softmax", loss="cross_entropy", verbos=False)
            self.client_list.append(client_i)
            
            
        ##-- For fine tune only
        if self.nz>0:
            ##---- 1. create loaders for fine tuning
            self.loaders_list_ft = []
            for path_i in path_list:
                loaders_i = self.make_loader_ft(path_i)
                self.loaders_list_ft.append(loaders_i)

            self.valid_loaders_ft = self.make_loader_ft(path_valid)
            self.test_loaders_ft = self.make_loader_ft(path_valid)
            
            
    def read_np_data(self, path):
        data = np.load(path)
        return data
        
    def make_loader(self, path):
        data = self.read_np_data(path)
        
        tensor_X = torch.tensor(data['x'])
        tensor_y = torch.tensor(data['y'])#.to(torch.int64)
        # tensor_z = torch.tensor(data['z'])
        
        dataset = TensorDataset(tensor_X, tensor_y)
        dataloader = DataLoader(dataset, batch_size=self.n_batch, shuffle=True)
        
        return dataloader
    
    def make_loader_ft(self, path):
        data = self.read_np_data(path)
        
        loader_list = []
        
        ## seperate loader for data with differert z
        for j in range(self.nz):
            idx_j = (data['z'] == j)
            
            tensor_x_j = torch.tensor(data['x'][idx_j])
            tensor_y_j = torch.tensor(data['y'][idx_j])
            
            dataset_j = TensorDataset(tensor_x_j, tensor_y_j)
            dataloader_j = DataLoader(dataset_j, batch_size=self.n_batch, shuffle=True)
            
            loader_list.append(dataloader_j)
            
        return loader_list
        
        
    def train(self, server_epoch, client_epoch):
        
        loss_list = []
        
        print("Training:")
        for i_epoch in range(server_epoch):
            print(f"Round {i_epoch}")
            
            ## sending out current central parameter
            param_current = self.server.get_central_param()
            for client in self.client_list:
                client.set_local_param(param_current)
            
            ## random sample of clients
            m_sel = max(1, int(self.c*self.n_client))
            sel_client_idx = np.random.choice(np.arange(self.n_client), size=m_sel, replace=False)
            
            print("  sel_client_idx = ", sel_client_idx)
            
            ## updating each client
            for idx in sel_client_idx:
                print(f"  Updating client {idx}...", end = '')
                client_i = self.client_list[idx]
                loader_i = self.loader_list[idx]
                # client_i.set_local_param(param_current)
                client_i.update_local(loader_i, client_epoch)
                print("Done.")
            
            ## updating server by federate average
            print(f"Updating server...", end = '')
            
            param_list = []
            weight_list = []
            for idx in sel_client_idx:
                client_i = self.client_list[idx]
                param_i, weight_i = client_i.get_local_param()
                param_list.append(param_i)
                weight_list.append(weight_i)

            #param_next = self.server.fed_avg(param_list, weight_list)
            #self.server.update_central(param_next)
            self.server.update_central(param_list, weight_list)
            
            print(f"Done.")
            
            
            ## validation
            valid_res = self.test(self.valid_loaders_ft)
            print("---- valid res ----")
            print(f"  acc = {valid_res['acc']}")
            print(f"  log_loss = {valid_res['log_loss']}")
            print(f"  count = {valid_res['count']}")
            print("\n")
            
            ## early stoping
            loss_current = valid_res['log_loss']
            if (len(loss_list)>0) and (loss_current>min(loss_list)*1.02):
                print("!!!! Eealy Stopping !!!!")
                break
            loss_list.append(loss_current)
            
            
    def train_ft(self, server_epoch, client_epoch):
        
        ## get the unfine-tuned parameters as initial
        param_0 = self.server.get_central_param()
        self.server.set_central_params_ft([param_0]*self.nz)
        
        loss_list = []
        
        print("Fine tuning:")
        for i_epoch in range(server_epoch):
            print(f"Round {i_epoch}") 
            
            ## sending out current central parameter
            param_ft_current = self.server.get_central_params_ft()
            for client in self.client_list:
                client.set_local_params_ft(param_ft_current)
            
            ## random sample of clients
            m_sel = max(1, int(self.c*self.n_client))
            sel_client_idx = np.random.choice(np.arange(self.n_client), size=m_sel, replace=False)
            
            print("  sel_client_idx = ", sel_client_idx)
            
            ## updating each client
            for idx in sel_client_idx:
                print(f"  Updating client {idx}...", end = '')
                client_i = self.client_list[idx]
                loaders_ft_i = self.loaders_list_ft[idx]
                # client_i.set_local_params_ft(param_ft_current)
                client_i.update_local_ft(loaders_ft_i, client_epoch)
                print("Done.")
                
            ## updating server by federate average
            print(f"Updating server...", end = '')
            
            params_list = []
            weights_list = []
            for idx in sel_client_idx:
                client_i = self.client_list[idx]
                params_i, weights_i = client_i.get_local_params_ft()
                params_list.append(params_i)
                weights_list.append(weights_i)

            self.server.update_central_ft(params_list, weights_list)
            
            print(f"Done.")
            
            ## validation
            valid_res_ft = self.test_ft(self.valid_loaders_ft)
            print("---- valid res ----")
            print(f"  acc = {valid_res_ft['acc']}, acc_z = {valid_res_ft['acc_z']}")
            print(f"  log_loss = {valid_res_ft['log_loss']}, log_loss_z = {valid_res_ft['log_loss_z']}")
            print(f"  count = {valid_res_ft['count']}, count_z = {valid_res_ft['count_z']}")
            print("\n")
        
            ## early stoping
            loss_current = valid_res_ft['log_loss']
            if (len(loss_list)>0) and (loss_current>min(loss_list)*1.02):
                print("!!!! Eealy Stopping !!!!")
                break
            loss_list.append(loss_current)

    def test(self, loaders):
        return self.server.test(loaders)
    
    def test_ft(self, loaders):
        return self.server.test_ft(loaders)

