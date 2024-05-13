import numpy as np
import torch


def relu(a):
    vec_relu = np.vectorize(lambda x: x * (x > 0))
    return vec_relu(a)
    

def attention_numpy_softmax(X, W):
    S = X @ W @ X.transpose() / np.sqrt(W.shape[0])
    S = torch.tensor(S)
    scores = np.array(torch.softmax(S, dim=1))
    output = scores @ X
    return output


def attention_softmax(X, W):
    S = X @ W @ X.t() / torch.sqrt(torch.tensor(W.shape[0], dtype=torch.float32))
    scores = torch.softmax(S, dim=1)
    output = scores @ X
    return output


def relu_attention_numpy(X, W, norm=True):
    S = X @ W @ X.transpose() / np.sqrt(W.shape[0])
    S = torch.tensor(S, dtype=torch.float32)
    S_relu = torch.nn.functional.relu(S)
    if norm:
        row_sum = torch.sum(S_relu, dim=1, keepdim=True)
        mask = (row_sum == 0).float()
        row_sum = row_sum + mask
        scores = np.array(S_relu / row_sum)
    else:
        scores = np.array(S_relu)
    output = scores @ X
    return output


def relu_attention(X, W, norm=True):
    S = X @ W @ X.t() / torch.sqrt(torch.tensor(W.shape[0], dtype=torch.float32))
    S_relu = torch.nn.functional.relu(S)
    if norm:
        row_sum = torch.sum(S_relu, dim=1, keepdim=True)
        mask = (row_sum == 0).float()
        row_sum = row_sum + mask
        scores = S_relu / row_sum
    else:
        scores = S_relu
    output = scores @ X
    return output


def rf_numpy(X, V):
    rf = relu(V @ X.flatten())
    return rf


def rf(X, V):
    X = X.view(-1)
    rf = torch.nn.functional.relu(V @ X)
    return rf


def drf_numpy(X, V):
    x = X.flatten()
    phi = x
    L = max(V.keys())
    for j in range(L):
        pre_act = V[j+1] @ phi
        phi = relu(pre_act)
    return phi


def drf(X, V):
    x = X.view(-1)
    phi = x
    L = max(V.keys())
    for j in range(L):
        pre_act = V[j+1] @ phi
        phi = torch.nn.functional.relu(pre_act)
    return phi


def project(xt, x0, max_distance):
    distance = torch.dist(xt, x0)
    if distance > max_distance:
        direction = (xt - x0) / distance
        xt.data = x0 + max_distance * direction