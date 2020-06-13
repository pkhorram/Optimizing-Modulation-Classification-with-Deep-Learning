import pickle
from utils import *

# Open this ./RML2016.10b/RML2016.10b.dat if you want to preprocess bigger dataset 
with open("RML2016.10a_dict.pkl", "rb") as p:
    d = pickle.load(p, encoding='latin1')

classes = []    
for i in d.keys():    
    if i[0] not in classes:
        classes.append(i[0])

# creating class dictionary for strings to digits transformation.
label_dict,  digit_label = digitizer(classes)

SNRs = {}
for key in d.keys():
    if key not in SNRs:
        SNRs[key[1]] = []
SNRs.keys()

j = 0
for keys in d.keys():
    for arrays in d[keys]:
        # convert labels to one-hot encoders.
        SNRs[keys[1]].append([onehot_encoder(label_dict, keys[0]),np.array(arrays)]) 

outfile = open('dataset','wb') # Save with other file name when using bigger data set (dataset_big)
pickle.dump(SNRs, outfile)
outfile.close()
