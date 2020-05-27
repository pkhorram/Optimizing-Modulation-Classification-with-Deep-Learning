import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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



def confusion_matrix_create (y_true, y_pred):
    
    cm = confusion_matrix(y_true,y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.show()


