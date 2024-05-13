import numpy as np
import argparse
import os
import time
import torch
import torch.optim as optim
import pickle
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--i')
parser.add_argument('--map')
parser.add_argument('--dataset')
args = parser.parse_args()

time.sleep(int(args.i))

save_dir = 'S_wed/' + args.dataset + '/' + args.map
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if args.map == 'raf':
    map = attention_softmax
    map_numpy = attention_numpy_softmax
elif args.map == 'raf_relu':
    map = relu_attention
    map_numpy = relu_attention_numpy
elif args.map == 'rf':
    map = rf
    map_numpy = rf_numpy
elif args.map == 'drf':
    map = drf
    map_numpy = drf_numpy

folder_name = 'imdb_dataset'  # your folder
with open(os.path.join(folder_name, 'new_embeddings_jan.pkl'), 'rb') as file:
    loaded_embeddings = pickle.load(file)

de = loaded_embeddings['X_emb'].shape[2]
ds = [192, 384, 768]

if 'raf' in args.map:
    ns = [10, 20, 50, 100, 150, 200, 250]
else:
    ns = [90, 100, 110, 120, 130, 140, 150]
    
if args.map == 'drf':
    Ls = [2, 4, 8]
    ds = [768]
    ns = [50, 60, 70, 80]
else:
    Ls = [1]

for d in ds:
    for n in ns:
        for L in Ls:
            D = d * n
            
            
            if args.map == 'rf':
                W = np.random.randn(D, D) / np.sqrt(D)
            elif args.map == 'drf':
                W = {}
                for j in range(L):
                    W[j+1] = np.sqrt(2) * np.random.randn(D, D) / np.sqrt(D)
            else:
                W = np.random.randn(d, d) / np.sqrt(d)

    
            if args.dataset == 'synthetic':
                X = np.random.randn(n, d)
            elif args.dataset == 'imdb':
                index = np.random.choice(loaded_embeddings['X_emb'].shape[0])
                sample = np.array(loaded_embeddings['X_emb'][index])
                X = sample[:n,:d]
                row_norms = np.linalg.norm(X, axis=1)
                X = X / row_norms[:, np.newaxis] * np.sqrt(d)
    
            print(n, flush=True)
            
            phi = map_numpy(X, W)
            phi = phi.flatten()  # doesn't do anything in rf
            
            
            xt = torch.tensor(X[0] + 0.1 * np.random.randn(d), requires_grad=True)
            X_1 = torch.tensor(X[1:])
            
            if args.map == 'drf':
                Wt = {}
                for j in range(L):
                    Wt[j+1] = torch.tensor(W[j+1])
            else:
                Wt = torch.tensor(W)
    
            x0 = xt.clone().detach()
            X_1_debug = X_1.clone().detach()
            
            
            loss_function = torch.nn.MSELoss()
    
            if 'rf' in args.map:
                lr = 100.
                num_steps = 300
            else:
                lr = 10.
                num_steps = 50 * n
            
            losses = []
            distances_xt = []
            distances_X_1 = []
            
            for step in range(num_steps):
                
                Xt = torch.cat((xt.unsqueeze(dim=0), X_1.detach()), dim=0)
                features = map(Xt, Wt)
                
                loss = - torch.sqrt(loss_function(features.view(-1), torch.tensor(phi, dtype=torch.float64)))
                loss.backward()
    
                with torch.no_grad():
                    xt.data = xt.data - lr * xt.grad.data
                    project(xt, x0, np.sqrt(d))
            
                xt.grad.zero_()
            
                distance_xt = torch.dist(xt, x0)
                distance_X_1 = torch.dist(X_1, X_1_debug)
    
                losses.append(loss.item())
                distances_xt.append(distance_xt.item())
                distances_X_1.append(distance_X_1.item())
    
                if step % 50 == 0:
                    print(loss.item(), np.linalg.norm(phi - features.view(-1).detach().numpy()) / np.linalg.norm(phi), distance_xt.item() / np.sqrt(d), flush=True)
                
            xd = xt.detach().numpy()
            XD = np.copy(X)
            XD[0] = xd
            
            phiD = map_numpy(XD, W)
            phiD = phiD.flatten()  # again nothing in rf
    
            S = np.linalg.norm(phi - phiD) / np.linalg.norm(phi)
        
            with open(os.path.join(save_dir, args.i + '.txt'), 'a') as f:
                f.write(str(d) + '\t' + str(n) + '\t' + str(L) + '\t' + str(np.linalg.norm(phi)) + '\t' + str(S) + '\n')
