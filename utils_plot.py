import numpy as np
import pandas as pd
import csv
import os
import re
from datasets import concatenate_datasets
import random
import pickle


def import_in_df_adversarial(folder, activation):

    data = []
    my_folder = folder + activation

    for filename in os.listdir(my_folder):
        if '.txt' in filename:
            with open(os.path.join(my_folder, filename), 'r') as f:
                reader = csv.reader(f,  delimiter='\t')
                for row in reader:
                    new_row = []
                    for j in range(3):
                        new_row.append(int(row[j]))
                    for j in range(3, 7):
                        new_row.append(float(row[j]))
                    data.append(new_row)

    df = pd.DataFrame(data=data, columns=(['n', 'd', 'N', 't0', 't1', 'D0', 'D1']))
    
    return df


def import_in_df_sensitivity_rf(folder, activation):

    data = []
    my_folder = folder + activation

    for filename in os.listdir(my_folder):
        if '.txt' in filename:
            with open(os.path.join(my_folder, filename), 'r') as f:
                reader = csv.reader(f,  delimiter='\t')
                for row in reader:
                    new_row = []
                    for j in range(3):
                        new_row.append(int(row[j]))
                    for j in range(3, 5):
                        new_row.append(float(row[j]))
                    data.append(new_row)

    df = pd.DataFrame(data=data, columns=(['d', 'n', 'L', 'norm', 'S']))
    
    return df
