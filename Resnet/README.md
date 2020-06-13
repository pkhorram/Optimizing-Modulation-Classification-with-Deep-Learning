# Optimizing-Modulation-Classification-with-Deep-Learning
This code implements the ResNet model given in the following paper, "Deep Neural Network Architectures for Modulation Classification". Reference: https://ieeexplore.ieee.org/document/8335483

Steps to run the code:

To run this code, download the tarfile from https://www.deepsig.ai/datasets (RADIOML 2016.10A for smaller dataset and RADIOML 2016.10B for bigger dataset)
Once you have this file, run "extract_tarfile.py" to extract the tar file and convert it to pickle format file.
Run "data_preprocessing.py" file to extract the data from pickle formatted file for preprocessing. This will generate a file called 'dataset' in case of smaller dataset or 'dataset_big' in case of bigger data set.
Once either of this file gets generated, run 'revised_data_preprocessing.py' if you generated smaller dataset or run 'revised_big_data_preprocessing.py' if you generated bigger dataset.
You will then have two files generated 'new_model_SNR_test_samples' and 'combined_SNR_data' if you ran 'revised_data_preprocessing.py' or you will have 'new_model_SNR_test_samples_bigdata' and 'combined_SNR_data_bigdata' if you ran 'revised_big_data_preprocessing.py'.
Now using these files, run 'revised_training.py' or 'revised_bigdata_training.py' depending on which dataset you want to use for training.
Once the model is trained, it will save accuracy, loss, best model and confusion matrices for each SNR value.
Once you have the results, you can run "load_and_plot_results.py" to generate the plots and confusion matrix.
There you go, you are successful in training a model and plotting the results!
Enjoy!
