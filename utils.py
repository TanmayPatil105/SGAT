import torch
import pickle as pkl
import sys

import networkx as nx
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import csv
import numpy as np
import scipy.sparse as sp

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')

class SaveToCSV:
    def __init__(self, file_name, ppi=False):
        self.file_name = file_name

        if ppi == True:
            row = ['Epoch', 'F1_score']
        else:
            row = ['Epoch', 'Train_accuracy', 'Valid_accuracy']

        with open (self.file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def add_ppi_row(self, epoch, f1_score):
        row = [epoch, f'{f1_score:.4f}']
        with open (self.file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def add_row(self, epoch, train_acc, valid_acc):
        row = [epoch, f'{train_acc:.4f}', f'{valid_acc:.4f}']
        with open (self.file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)
