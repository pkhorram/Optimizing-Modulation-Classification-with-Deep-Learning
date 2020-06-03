# Improvements to Modulation Classification using Deep Learning

This code implements the CLDNN model given in the following paper, **"Deep Neural Network Architectures for Modulation Classification"**.
Reference: https://ieeexplore.ieee.org/document/8335483

Steps to run the code:

1. To run this code, download the tarfile from https://www.deepsig.ai/datasets (RADIOML 2016.10A)
2. Once you have this file, run **"extract_tarfile.py"** to extract the tar file and convert it to pickle format file.
3. Run **"data_preprocessing.py"** file to extract the data from pickle formatted file for preprocessing. This will generate a file called **'dataset'**.
4. Now, you are ready to train your CLDNN Model. Just run **"training.py"** and it will start training the code and will save accuracy, loss, best model and confusion matrices for each SNR value.


