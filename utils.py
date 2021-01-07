import tensorflow as tf
import numpy as np


def test_data(lb,dataset):
    num_sample1=len(dataset)//2

    num_sample=len(dataset)
    #process and batch the 2test set(one for local model and one for global)
    label=[]
    data=[]
    dataset=list(dataset)
    for i in range(num_sample1):
        label.append(dataset[i][1])
        img=np.transpose(dataset[i][0].numpy())
        data.append(img) 

    label=lb.fit_transform(label)
    test_1 = tf.data.Dataset.from_tensor_slices((list(data), list(label))).batch(num_sample1)

    label=[]
    data=[]
    for i in range(num_sample1,num_sample):
        label.append(dataset[i][1])
        img=np.transpose(dataset[i][0].numpy())
        data.append(img)

    label=lb.fit_transform(label)
    test_2 = tf.data.Dataset.from_tensor_slices((list(data), list(label))).batch(num_sample)
    
    return test_1,test_2

def batch_data(lb,img_indxs,dataset_train):
    bs=len(img_indxs)
    label=[]
    img=[]
    for i in img_indxs:
        label.append(dataset_train[i][1])
        img.append(dataset_train[i][0])    
    label=lb.fit_transform(label)
    dataset = tf.data.Dataset.from_tensor_slices((list(img),list(label)))
    return dataset.shuffle(len(label)).batch(bs)

