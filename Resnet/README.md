# Improvements to Modulation Classification using Deep Learning

This code implements the ResNet model given in the following paper, **"Deep Neural Network Architectures for Modulation Classification"**.
Reference: https://ieeexplore.ieee.org/document/8335483

Steps to run the code:

1. To run this code, download the tarfile from https://www.deepsig.ai/datasets (RADIOML 2016.10A for smaller dataset and RADIOML 2016.10B for bigger dataset)
2. Once you have this file, run **"extract_tarfile.py"** to extract the tar file and convert it to pickle format file.
3. Run **"data_preprocessing.py"** file to extract the data from pickle formatted file for preprocessing. This will generate a file called **'dataset'** in case of smaller dataset or **'datasetB'** in case of bigger data set.
4. Once either of this file gets generated, run **'revised_data_preprocessing.py'** if you generated smaller dataset or run **'revised_data_preprocessing_B.py'** if you generated bigger dataset. 
5. You will then have two files generated **'new_model_SNR_test_samples'** and **'combined_SNR_data'** if you ran **'revised_data_preprocessing.py'** or you will have **'new_model_SNR_test-B'** and **'combined_SNR_data-B'** if you ran **'revised_data_preprocessing_B.py'**. 
6. Now using these files, run **'revised_training.py'** or **'revised_training-B.py'** depending on which dataset you want to use for training. 
7. Once the model is trained, it will save accuracy, loss, best model and confusion matrices for each SNR value.
8. Once you have the results, you can use **"Plots.ipynb"** to generate the plots and confusion matrix. 
9. There you go, you are successful in training a model and plotting the results!


Enjoy!
