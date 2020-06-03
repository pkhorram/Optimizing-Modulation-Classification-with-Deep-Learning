import numpy as np


def digitizer(labels):
    
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
    
            
def onehot_encoder(L_dict,  labels):
    num_classes = len(L_dict)
    vector = np.zeros(num_classes)
    vector[L_dict[labels]-1] = 1
    return vector



