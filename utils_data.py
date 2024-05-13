import numpy as np
import pandas as pd
import csv
import os
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import concatenate_datasets
import random
import pickle


def generate_embeddings(my_dataset, batch_size, model, layer, device):
    X_train_tok = my_dataset['X_train_tok']
    X_test_tok = my_dataset['X_test_tok']
    embeddings_dict = {'layer': layer}

    for (key, X_tok) in [('X_train_emb', X_train_tok), ('X_test_emb', X_test_tok)]:
        print('Generating {}'.format(key), flush=True)
        input_ids, token_type_ids, attention_mask = X_tok.values()
        tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask)
        train_loader = DataLoader(tensor_dataset, batch_size=batch_size)
        embedding_size = 768
        context_length = len(X_tok['input_ids'][0])
        
        embeddings = torch.empty(0, context_length, embedding_size)
        batch_id = 0
        for batch in train_loader:
            batch_id += 1
            if batch_id % 10 == 0:
                print('Batch id = {}'.format(batch_id), flush=True)
            batch_input_ids, batch_token_type_ids, batch_attention_mask = batch
            batch_input_ids= batch_input_ids.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            with torch.no_grad():
                output = model(batch_input_ids, token_type_ids=batch_token_type_ids, attention_mask=batch_attention_mask)
            layers = output[2]
            my_layer = layers[int(layer)].to('cpu')
            embeddings = torch.cat((embeddings, my_layer), dim=0)
            torch.cuda.empty_cache()

        embeddings_dict[key] = embeddings

    return embeddings_dict



def create_imdb_dataset(model, dataset, train_size, test_size, tokenizer, device):
    ip = 0
    im = 0
    X = []
    y = []
    
    for row in concatenate_datasets([dataset['train'], dataset['test']]):
        label = row['label']
        text = row['text']
        num_tokens = len(tokenizer.tokenize(text))
        
        if num_tokens < 240:
            if ip + im < train_size + test_size:
                if label == 1 and ip < (train_size + test_size) // 2:
                    X.append(text)
                    y.append(1)
                    ip += 1
                if label == 0 and im < (train_size + test_size) // 2:
                    X.append(text)
                    y.append(-1)
                    im += 1
        if im + ip >= train_size + test_size:
            break

    combined_lists = list(zip(X, y))
    random.shuffle(combined_lists)
    shuffled_X, shuffled_y = zip(*combined_lists)

    X_train = list(shuffled_X)[:train_size]
    y_train = list(shuffled_y)[:train_size]
    
    X_test = list(shuffled_X)[train_size:train_size+test_size]
    y_test = list(shuffled_y)[train_size:train_size+test_size]
    
    X_train_tok = tokenizer(X_train, padding='max_length', truncation=True, return_tensors='pt', max_length=256)
    X_test_tok = tokenizer(X_test, padding='max_length', truncation=True, return_tensors='pt', max_length=256)
    
    X_train_tok = {key: value.to(device) for key, value in X_train_tok.items()}
    X_test_tok = {key: value.to(device) for key, value in X_test_tok.items()}

    y_train = torch.tensor(y_train).to(device)
    y_test = torch.tensor(y_test).to(device)

    my_dataset = {'X_train': X_train, 'X_train_tok': X_train_tok, 'y_train': y_train, 'X_test': X_test, 'X_test_tok': X_test_tok, 'y_test': y_test}

    return my_dataset

