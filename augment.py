#!/usr/bin/env python
# coding: utf-8

from torchvision import datasets, transforms
import numpy as np
import cv2
from tqdm import tqdm
import torch 

img_sz=0
def basic(image_size):
    #defining the transformation method
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize((0.5), (0.5,)),
    ])
    img_sz=image_size
    return transform


transform_train_minor_h = transforms.Compose([
            transforms.RandomPerspective(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-45,45), fill=(0,)),
            ])

transform_train_minor_nh = transforms.Compose([
            transforms.RandomPerspective(),
            transforms.RandomRotation(degrees=(-45, 45), fill=(0,)),
            ])
            


def add_augment_image(dict_users,dataset,idx,num_user,s=4):
    count=len(dataset)
    final_idx=len(dict_users[0])
    for j in tqdm(range(num_user)):
        for i in dict_users[j][idx:final_idx]:
            
            label=dataset[i][1]   
            
            #blurring the datatset
            img1=cv2.blur(dataset[i][0].numpy(),(2,2))
            img1=torch.from_numpy(img1)
            dataset.append([img1,label])  
            dict_users[j] = np.append(dict_users[j], count)
            count += 1

            for k in range(s-1):
                img2=transform_train_minor_h(dataset[i][0])  #using the trasformation technique
                dataset.append([img2,label])   
                dict_users[j] = np.append(dict_users[j], count)
                count += 1
                
    print('Added '+str(s)+' Augemented Images/class for each user to dataset')      
    return dict_users,dataset

#add your new augmentataion methods in this file