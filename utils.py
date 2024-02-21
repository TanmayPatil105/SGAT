import torch
import pickle as pkl
import sys

import networkx as nx
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import csv
import numpy as np
import scipy.sparse as sp
import pandas as pd
from matplotlib import pyplot as plt

def debug():
    torch.set_printoptions(profile="full")

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

class LineGraph:
    def __init__(self, dataset, ppi=False):
        plt.rcParams["figure.figsize"] = [28.00, 14.00]
        plt.rcParams["figure.autolayout"] = True

        if ppi == True:
            cols = ["Epoch", "F1_score"]
            dataset = pd.read_csv("ppi-accuracy.csv", usecols=cols)
            plt.plot(dataset.Epoch, dataset.F1_score, label='F1 score')
            plt.title("F1 score vs Epoch")
            plt.xlabel("No. of Epochs")
            plt.ylabel("F1 score")

            plt.savefig ("ppi-graph.png")
        else:
            cols = ["Epoch", "Train_accuracy", "Valid_accuracy"]
            dataset = pd.read_csv(dataset + "-accuracy.csv", usecols=cols)
            plt.plot(dataset.Epoch, dataset.Train_accuracy, label='Training accuracy')
            plt.plot(dataset.Epoch, dataset.Valid_accuracy, label='Validation accuracy')
            plt.title("Accuracy vs Epoch")
            plt.xlabel("No. of Epochs")
            plt.ylabel("Accuracy")

            plt.savefig (dataset + "-graph.png")

        # On Jypter notebook
        # plt.legend()
        # plt.show()

def create_masks(size, percentages):
    """
    Splits dataset into three parts:
        - Train mask
        - Validation mask
        - Test mask
    """
    masks = []
    start = 0

    for percent in percentages:
        end = start + int(size * percent)
        tensor = torch.zeros(size)
        tensor[start:end] = 1
        masks.append(tensor)
        start = end

    return masks[0], masks[1], masks[2]
