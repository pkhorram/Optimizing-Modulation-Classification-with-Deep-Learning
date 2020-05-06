import numpy as np
import random 
import matplotlib.pyplot as plt



def digtizer(labels):
    
    unique_labels = np.unique(labels)
    label_dict = {}
    num = 1
    for i in unique_labels:
        label_dict[i] = num
        num += 1 
    
    digit_label = []
    for i in labels:
        digit_label.append(label_dict[i])
    
    return label_dict,  digit_label 
    
            
def onehot_encoder(num_class,  labels):
    
    vector = np.zeros((len(labels), num_class))
    for i in range(len(labels)):
        vector[i,labels[i]-1] = 1
    return vector



