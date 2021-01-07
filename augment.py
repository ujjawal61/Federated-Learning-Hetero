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
            


def normal_aug(dict_users,dataset,idx,num_user,s=4):
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


#using image augmentation twice and add 4 transformed image
def get_random_eraser(input_img, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    img_c, img_w, img_h = input_img.shape
    temp=input_img.numpy().copy()
    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break
            
    c = np.random.uniform(v_l, v_h)
    temp[:,top:top + h, left:left + w] = c

    return torch.from_numpy(temp)

def advanced_aug(dict_users,dataset,idx,num_user,s=4):
    count=len(dataset)
    final_idx=len(dict_users[0])
    for j in tqdm(range(num_user)):
        for i in dict_users[j][idx:final_idx-1]:
            label=dataset[i][1]   
            
            #adding 2 weighted image
            img1=dataset[i][0]
            img2=dataset[i+1][0]
            img3= 0.8*img1+0.2*img2
            dataset.append([img3,label])   
            dict_users[j] = np.append(dict_users[j], count)
            count += 1
            
            for k in range(s-1):
            #random removing block
                img4= get_random_eraser(img1) 
                dataset.append([img4,label])   
                dict_users[j] = np.append(dict_users[j], count)
                count += 1
                
    print('Added '+str(s)+' Augemented Images/class for each user to dataset') 
    return dict_users,dataset
#add your new augmentataion methods in this file