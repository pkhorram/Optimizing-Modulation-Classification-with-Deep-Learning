# Optimizing-Modulation-Classification-with-Deep-Learning


This code implements the Robust CNN model given in the following paper, **"Robust and Fast Automatic Modulation
Classification with CNN under Multipath Fading
Channels"**.
Reference: https://arxiv.org/pdf/1911.04970.pdf

Steps to run the code:

1. To run this code, download the tarfile from https://www.deepsig.ai/datasets (RADIOML 2016.10A for smaller dataset and RADIOML 2016.10B for bigger dataset)
2. Once you have this file, run **"extract_tarfile.py"** to extract the tar file and convert it to pickle format file.
3. Run **"data_preprocessing.py"** file to extract the data from pickle formatted file for preprocessing. This will generate a file called **'dataset'**.
4. Once either of this files get generated, run **'revised_data_preprocessing.py'** This will rearange the dataset by merging are data points from all classes and SNR values together.
5. Running **'revised_data_preprocessing.py'**, You will then have two files generated **'new_model_SNR_test_samples'** which includes all the test data for each SNR value (This will later be used to evaluate the performance of the trained model for each SNR value), and **'combined_SNR_data'** which (containes the training set).   
6. Now using these files, run **'revised_training.py'*
7. Once the model is trained, it will save accuracy, loss, best model and confusion matrices for each SNR value.
8. Once you have the results, you can run **"load_and_plot_results.py"** to generate the plots and confusion matrix. 






