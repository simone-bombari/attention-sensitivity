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
parser.add_argument('--optim')
parser.add_argument('--training')
parser.add_argument('--dataset')
args = parser.parse_args()

time.sleep(int(args.i))

save_dir = os.path.join('./gen_wed_1_manyN_new', args.dataset, args.map, args.training, args.optim)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if args.map == 'raf':
    map = attention_softmax
    map_numpy = attention_numpy_softmax
    ns = [40, 120]
elif args.map == 'raf_relu':
    map = relu_attention
    map_numpy = relu_attention_numpy
    ns = [120]
elif args.map == 'rf':
    map = rf
    map_numpy = rf_numpy
    ns = [40]

iterations = 1

Ns = np.arange(1100, 1301, 100)

folder_name = 'imdb_dataset'
with open(os.path.join(folder_name, 'new_embeddings_jan.pkl'), 'rb') as file:
    loaded_embeddings = pickle.load(file)

d = loaded_embeddings['X_emb'].shape[2]  # 768
penalty = 0

for j in range(iterations):  
    for N in Ns:
        for n in ns:
            D = d * n
            k = D
    
            if args.dataset == 'synthetic':
                Xs = np.random.randn(N, n, d)
                u = np.random.randn(D) / np.sqrt(D)
                Pu = np.outer(u, u)
                Y = np.sign(Xs.reshape(Xs.shape[0], -1) @ u)
                
                x = u + (np.eye(D) - Pu) @ np.random.randn(D)  # A test sample with ground truth label 1
                y = np.sign(np.dot(x, u))  # = 1
                x = x.reshape((n, d))
                
            elif args.dataset == 'imdb':
                indices = np.random.choice(loaded_embeddings['X_emb'].shape[0], size=N+1, replace=False)
                samples = np.array(loaded_embeddings['X_emb'][indices[1:]])
                Y = np.array(loaded_embeddings['y'][indices[1:]])
                Xs = samples[:,:n,:d]
                row_norms = np.linalg.norm(Xs, axis=2)
                Xs = Xs / row_norms[:, :, np.newaxis] * np.sqrt(d)
                
                x = np.array(loaded_embeddings['X_emb'][indices[0]])
                y = np.array(loaded_embeddings['y'][indices[0]])
                x = x[:n,:d]
                row_norms = np.linalg.norm(x, axis=1)
                x = x / row_norms[:, np.newaxis] * np.sqrt(d)
            
            if args.map == 'rf':
                W = np.random.randn(k, D) / np.sqrt(D)
            else:
                W = np.random.randn(d, d) / np.sqrt(d)
            
            Phi = None
            for X in Xs:
                phi = map_numpy(X, W)
                phi = phi.flatten()  # doesn't do anything in rf
                if Phi is None:
                    Phi = [phi]
                else:
                    Phi = np.concatenate([Phi, [phi]], axis=0)
    
            Phip = np.linalg.pinv(Phi)
            theta = Phip @ Y
            P_o = np.identity(D) - Phip @ Phi
            
            phi = map_numpy(x, W)
            phi = phi.flatten()  # doesn't do anything in rf

            if args.training == 'total':
                Phi1 = np.concatenate([Phi, [phi]], axis=0)
                Y1 = np.append(Y, y)
                theta1 = np.linalg.pinv(Phi1) @ Y1
            elif args.training == 'fine_tuning':
                theta1 = theta + phi * (y - np.inner(phi, theta)) / np.linalg.norm(phi) ** 2
            
            xt = torch.tensor(x[0], requires_grad=True)
            X_1 = torch.tensor(x[1:])
            W = torch.tensor(W)
    
            x0 = xt.clone().detach()
            X_1_debug = X_1.clone().detach()
            
            target = torch.tensor(-y).double()
            
            Po_phi = torch.tensor(P_o @ phi).clone().detach()
            Po_phi = Po_phi / torch.norm(Po_phi, p=2) ** 2

            just_phi = torch.tensor(phi).clone().detach()
            just_phi = just_phi / torch.norm(just_phi, p=2) ** 2
            
            loss_function = torch.nn.MSELoss()
            
            if 'rf' in args.map:
                lr = 100.
                num_steps = 300
            else:
                lr = 1.
                num_steps = 50 * n
            
            losses = []
            distances_xt = []
            distances_X_1 = []
            
            for step in range(num_steps):
                
                Xt = torch.cat((xt.unsqueeze(dim=0), X_1.detach()), dim=0).double()
                features = map(Xt, W)
                
                if args.optim == 'features':
                    out = torch.dot(features.view(-1), just_phi_n.view(-1).double())
                    loss = loss_function(out, torch.tensor(-1).double())
                elif args.optim == 'feat_align':
                    out = torch.dot(features.view(-1), Po_phi.view(-1).double())
                    loss = loss_function(out, torch.tensor(-1).double())
                elif args.optim == 'loss':
                    out = torch.dot(features.view(-1), torch.tensor(theta1))
                    loss = loss_function(out, target)

                if penalty > 0:
                    pen_loss = torch.dot(features.view(-1) - just_phi.view(-1).double(), torch.tensor(theta).double()) ** 2
                    loss = loss + penalty * pen_loss
                    
                loss.backward()
    
                with torch.no_grad():
                    xt.data = xt.data - lr * xt.grad.data
                    project(xt, x0, np.sqrt(d))
            
                xt.grad.zero_()
            
                distance_xt = torch.dist(xt, x0)
                distance_X_1 = torch.dist(X_1, X_1_debug)
                
                XD = Xt.clone().detach()
                phiD = map(XD, W).numpy().flatten()
                D0 = phiD.transpose() @ theta
                t0 = phi.transpose() @ theta

                print(distance_xt.item(), loss.item(), D0, t0, flush=True)

                
            xd = xt.detach().numpy()
            XD = np.copy(x)
            XD[0] = xd
            
            phiD = map_numpy(XD, np.array(W))
            phiD = phiD.flatten()  # again nothing in rf
    
            D0 = phiD.transpose() @ theta
            D1 = phiD.transpose() @ theta1
            t0 = phi.transpose() @ theta
            t1 = phi.transpose() @ theta1
        
            with open(os.path.join(save_dir, args.i + '.txt'), 'a') as f:
                f.write(str(n) + '\t' + str(d) + '\t' + str(N) + '\t' + str(t0) + '\t' + str(t1) + '\t' + str(D0) + '\t' + str(D1) + '\n')
