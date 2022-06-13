# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 19:22:37 2022

@author: Windows User
"""

import torch
import torch.utils.data as data
import os
import numpy as np
import shutil
import fnmatch
import glob
from torchvision import transforms as T
from numpy import random as nr

""" param """



def sample_data(data_path, species, mode):
    path_feature = data_path + mode + '-' + species + '-31_feature.npy' 
    path_label = data_path + mode + '-' + species + '-31_label.npy' 

    feature_all = np.load(path_feature)
    label_all = np.load(path_label) 
    n = label_all.shape[0]
    X=torch.Tensor(n,1,31,21)
    Y=torch.Tensor(n,2)

    inds=torch.randperm(n)
    for i,index in enumerate(inds):
        x=feature_all[index]
        y=label_all[index]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        X[i]=x
        Y[i]=y
    return X,Y


def create_target_samples(data_path, species, mode):
    path_feature = data_path + mode + '-' + species + '-31_feature.npy' 
    path_label = data_path + mode + '-' + species + '-31_label.npy' 

    feature_all = np.load(path_feature)
    label_all = np.load(path_label) 
    n = label_all.shape[0]
    X=torch.Tensor(n,1,31,21)
    Y=torch.Tensor(n,2)


    inds=torch.randperm(n)
    for i,index in enumerate(inds):
        x=feature_all[index]
        y=label_all[index]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        X[i]=x
        Y[i]=y
    return X,Y



def create_groups(X_s,Y_s,X_t,Y_t,seed=1):
    #change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)
    
    n = X_t.shape[0] #10*shot
    Y_t_one = Y_t[:,0]
    Y_s_one= Y_s[:,0]
    num1 = Y_t_one.sum(-1)
    num0 = n-num1
    shot = int(min(num1,num0))
    
    n=X_s.shape[0]
    num1 = Y_s_one.sum(-1)
    num0 = n-num1
    min_s = int(min(num1,num0))
    
    #shuffle order
    classes = torch.unique(Y_t_one)
    classes = classes[torch.randperm(len(classes))]

    class_num = classes.shape[0]




    def s_idxs(c):
        idx=torch.nonzero(Y_s_one.eq(int(c)))
        idx[torch.randperm(len(idx))]
        idx = idx[0:min_s]
        return idx.squeeze()
        
    def t_idxs(c):
        idx=torch.nonzero(Y_t_one.eq(int(c)))
        idx[torch.randperm(len(idx))]
        idx = idx[0:shot]
        return idx.squeeze()


    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))


    source_matrix=torch.stack(source_idxs)

    target_matrix=torch.stack(target_idxs)


    G1, G2, G3, G4 = [], [] , [] , []
    Y1, Y2 , Y3 , Y4 = [], [] ,[] ,[]


    for i in range(2):
        for j in range(int(shot/2)):
            G1.append((X_s[source_matrix[i][j*2]],X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]],Y_s[source_matrix[i][j*2+1]]))
            G2.append((X_s[source_matrix[i][j]],X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i][j]]))
            G3.append((X_s[source_matrix[i%2][j]],X_s[source_matrix[(i+1)%2][j]]))
            Y3.append((Y_s[source_matrix[i % 2][j]], Y_s[source_matrix[(i + 1) % 2][j]]))
            G4.append((X_s[source_matrix[i%2][j]],X_t[target_matrix[(i+1)%2][j]]))
            Y4.append((Y_s[source_matrix[i % 2][j]], Y_t[target_matrix[(i + 1) % 2][j]]))



    groups=[G1,G2,G3,G4]
    groups_y=[Y1,Y2,Y3,Y4]


    return groups,groups_y


def sample_groups(X_s,Y_s,X_t,Y_t,seed=1):


    print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,seed=seed)



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
            self.path_feature = self.next_input[0]
            self.path_label = self.next_input[1]

        except StopIteration:
            self.next_input = None
            self.path_feature = None
            self.path_label = None
            return

            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input 
        feature = input[0]
        label = input[1]
        self.preload()
        return feature,label
    

class dataset(data.Dataset):
    def __init__(self,data_path, species, mode = 'train'):
        self.mode = mode
        self.data_path = data_path
        self.species = species

        if self.mode == 'train':
            path_feature = self.data_path + self.mode + '-' + self.species + '-31_feature.npy'            
            self.path_feature = path_feature
            path_label = self.data_path + self.mode + '-' + self.species + '-31_label.npy'            
            self.path_label = path_label
            self.features, self.labels = self.augmentation(rotation = True, flipping = True)
            
        if self.mode == 'test':
            path_feature = self.data_path + self.mode + '-' + self.species + '-31_feature.npy'            
            self.path_feature = path_feature
            path_label = self.data_path + self.mode + '-' + self.species + '-31_label.npy'            
            self.path_label = path_label
            self.features, self.labels = self.augmentation(rotation = True, flipping = True)
             
        if self.mode == 'valid':
            path_feature = self.data_path + self.mode + '-' + self.species + '-31_feature.npy'            
            self.path_feature = path_feature
            path_label = self.data_path + self.mode + '-' + self.species + '-31_label.npy'            
            self.path_label = path_label
            self.features, self.labels = self.augmentation(rotation = True, flipping = True)
            
            
    def augmentation(self, rotation = False, flipping = False):
        features, labels= [], []
        feature_path = self.path_feature
        
        label_path = self.path_label
        feature_all = np.load(feature_path)
        
        label_all = np.load(label_path) 
        
        for i in range(label_all.shape[0]):
            features.append(feature_all[i])
            labels.append(label_all[i])
        return features, labels
    
    def __getitem__(self, index):
 
        features = self.features

        labels = self.labels

        feature_singel = features[index]
        label_singel = labels[index]
        return feature_singel, label_singel
    
    def __len__(self):
        return len(self.features)
    
    

