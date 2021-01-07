#!/usr/bin/env python
# coding: utf-8

import numpy as np

#this function is sampling the cifar image index in non iid way, each client will have majority of 2 class and 8 minor class
def sampling(dataset, num_users, p):

    idxs = np.arange(len(dataset),dtype=int)
    labels = np.array(dataset.targets)
    label_list = np.unique(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    #print(idxs_labels)
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    n_data=int(len(dataset)/(3*num_users))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    #Sample majority class for each user
    user_majority_labels = []
    for i in range(num_users):
        majority_labels = np.random.choice(label_list, 2, replace = False) #2 represent the numbers of majority classes each user will have
        user_majority_labels.append(majority_labels)
        #label_list = list(set(label_list) - set(majority_labels))
        #print(label_list)
        print(i,'\t',majority_labels)
        majority_label_idxs = (majority_labels[0] == labels[idxs])| (majority_labels[1] == labels[idxs])
        
        sub_data_idxs = np.random.choice(idxs[majority_label_idxs], int(p*n_data), replace = False)
        
        dict_users[i] = np.concatenate((dict_users[i],sub_data_idxs))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs)))
        
        #assigning minor classes to each client
    if(p < 1.0):
        for i in range(num_users):
            majority_labels = user_majority_labels[i]
            
            non_majority_label_idxs = (majority_labels[0] != labels[idxs]) | (majority_labels[1] != labels[idxs])
            
            sub_data_idxs = np.random.choice(idxs[non_majority_label_idxs], int((1-p)*n_data), replace = False)
            
            dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs))
            idxs = np.array(list(set(idxs) - set(sub_data_idxs)))
    idx=int((p)*n_data)
    
    return dict_users,idx