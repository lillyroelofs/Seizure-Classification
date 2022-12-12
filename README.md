## Seizure-Classification - Lilly Roelofs - 11/30/2022

## Synopsis 
This brief project was for one of my senior level biomedical engineering courses. We were asked to perform data analysis and modeling on electrophysiological signal data. 
I chose to use the [EEG Seizure Analysis Dataset](https://www.kaggle.com/datasets/adibadea/chbmitseizuredataset) on the Kaggle website, which is a subset of the 
original [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/chb21/#files-panel). After importing the data and doing some exploratory analysis, I 
applied the continuous wavelet transform (CWT) to the signals. From here, I pushed the transformed signals through a shallow neural network. The final accuracy for 
training is 88.65% and for validation is 67.84%. Due to time constraints I was not able to revise this methodology or tune the parameters, although it is clear
that improvement is needed from 1) the low validation accuracy, and 2) the large gap between the training and validation accuracies (indicating overfitting). 

An interesting point I noticed during the modeling was that less information (i.e. less dimensions for the CTW, less layers and nodes for the NN) increased the 
classification accuracy. I have generally operated under the assumption that the more information you can extract, the better the performance. However, I am suspecting 
that the additional information was capturing “irrelevant” features and therefore negatively informing the model. Of course this is just an observation, there definitely
could be other factors at play here. 

## Description of files/folders in Seizure Classification Repo

**project2_seizure_classification.py** - This is the python script I developed for the project. 

**Project 2 Report - GitHub Version.pdf** - This is the report I wrote for the assignment. There are many details addressed here regarding the code and data. 



