import random
import argparse
import os
import pickle
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from utils_data import create_imdb_dataset, generate_embeddings
from datasets import load_dataset

dataset = load_dataset('imdb')
train_size = 100
test_size = 0

model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, config=config)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
model.to(device)
print(device)

folder_name = 'imdb_dataset_debug'

if os.path.exists(folder_name):
    with open(os.path.join(folder_name, 'data.pkl'), 'rb') as file:
        my_dataset = pickle.load(file)
else:
    my_dataset = create_imdb_dataset(model, dataset, train_size, test_size, tokenizer, device)
    os.makedirs(folder_name)
    with open(os.path.join(folder_name, 'data.pkl'), 'wb') as file:
        pickle.dump(my_dataset, file)


save_embeddings = True
if save_embeddings:
    
    batch_size = 20
    embeddings = generate_embeddings(my_dataset, batch_size, model, 0, device)

    some_embeddings = {'X': my_dataset['X_train'],
    'X_tok': my_dataset['X_train_tok'],
    'X_emb': embeddings['X_train_emb'],
    'y': my_dataset['y_train']}

with open(os.path.join(folder_name, 'embeddings.pkl'), 'wb') as file:
    pickle.dump(some_embeddings, file)
