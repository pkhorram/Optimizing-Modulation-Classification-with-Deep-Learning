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



def confusion_matrix_create (y_true, y_pred, labels_dict, title):
    
    labels = []
    for i in labels_dict.items():
        labels.append(i[0])
    y_true = np.argmax(y_true, axis =1)
    y_true = np.array(y_true) + 1
    y_pred = np.array(y_pred) + 1
    
    
    
    updated_pred = []
    updated_true = []

    for i in range(len(y_true)):

        for key,value in labels_dict.items():
            if value == y_true[i]:
                updated_true.append(key)

            if value == y_pred[i]:
                updated_pred.append(key)

    cm = confusion_matrix(updated_true,updated_pred, labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.xticks(ticks=[-1,0,1,2,3,4,5,6,7,8,9,10], rotation=45)
    plt.yticks(ticks=[-1,0,1,2,3,4,5,6,7,8,9,10], rotation=45)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    ax.set (title=title, 
            ylabel='True label',
            xlabel='Predicted label')
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.show()
