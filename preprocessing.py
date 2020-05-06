import numpy as np
import pickle

with open("RML2016.10a_dict.pkl", "rb") as p:
    d = pickle.load(p, encoding='latin1')


t = 0
for i in d.keys():
    print('The shape of key', i, 'is:', d[('QPSK', 2)].shape )
    t+=1
print('The total number of keys:', t)


L = []
for i in d.keys():
        L.append(i[0])
        

unique_modules  = np.unique(L)
print('total number of modulations:', len(unique_modules))
num = 1
for i in unique_modules:
    print('Class', num, ':', i)
    num +=1

new_data = {}
for keys in unique_modules:
    new_data[keys] = []
           
           
for u_keys in unique_modules:
    for keys in d:
        if keys[0] == u_keys:
            new_data[u_keys].append(d[keys])
            
for key in new_data:
    shape = np.array(new_data[key]).shape
    new_data[key] = np.reshape(np.array(new_data[key]), [shape[0]*shape[1],shape[2],shape[3]])
    np.random.shuffle(new_data[key])
    
    
    
reduced_data = {}
for keys in unique_modules:
    reduced_data[keys] = []
    
for keys in reduced_data:
    reduced_data[keys] = new_data[keys][:2000,:,:]
    
dataset = []
for keys in reduced_data:
    for data in reduced_data[keys]:
        dataset.append((keys, data))
    
    
pickle_file = open('dataset.pickle', 'wb')
pickle.dump(np.array(dataset), pickle_file)
pickle_file.close()
    
    
print('Done !')
print('data shape for each class in now:',  reduced_data[next(iter(reduced_data))].shape)