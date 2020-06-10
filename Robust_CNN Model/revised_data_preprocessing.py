import numpy as np
import pickle 
import itertools
from random import shuffle

with open('./dataset', 'rb') as file:
    data = pickle.load(file,encoding = 'Latin')
    
    

for key in data.keys():  
    shuffle(data[key])
    


new_data = {'combined':[]}
SNR_test = {}


for key in data.keys():
       
    train_len = int(0.9*len(data[key]))    
    new_data['combined'].append(data[key][:train_len])
    SNR_test[key] = data[key][train_len:]
    
    
new_data['combined'] = list(itertools.chain.from_iterable(new_data['combined']))   
    

outfile = open('new_model_SNR_test_samples','wb')
pickle.dump(SNR_test,outfile)
outfile.close()


outfile = open('combined_SNR_data','wb')
pickle.dump(new_data,outfile)
outfile.close()