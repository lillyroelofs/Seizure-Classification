# -*- coding: utf-8 -*-
"""
Lilly Roelofs - 1875385 
Project 2: Data Analysis & Modeling 
BIOE 4308 - Dr. Romero-Ortega
Classifying seizure occurence 
"""

# ----------------------------------------------------------------------------
## Importing libraries & defining variables
import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
class_labels = ['Normal State', 'Seizure']



# ----------------------------------------------------------------------------
## Defining functions 


# Printing information for each of the folders, i.e. their size and contents
def print_info(npz_name, npz_var, signals, label, contains_labels):
    print('\nParameters of {}\nFiles stored in the numpy zip: {}'.format(npz_name, npz_var.files))
    print('Size of {}:'.format(npz_var.files[0]), signals.shape)
    if contains_labels == 1:
        print('Size of {}:'.format(npz_var.files[1]), label.shape)
        # Measurement of class label balance (percent of seizure vs normal state)
        print('Percentage of seizure occurences: {}%'.format(str(round((np.count_nonzero(label == 1)/len(label))*100, 3)))) # number of "1" entries / total number of entries 
    
    
# Plotting all channels from a single instance on top of each other 
def plot_together(var_of_interest, sample_num, labels, train):
    for num in sample_num: # for each instance in the provided list
        plt.figure() # open a matplotlib figure
        for i in range(var_of_interest.shape[1]): # for each channel in this instance 
            plt.plot(var_of_interest[num, i, :], color='black') # plot each channel on top of one another -- [instance #, channel #, entire wave (all 256 points)]
        classification = class_labels[labels[num]] # determine the classification of this instance 
        str1 = 'EEG {} Sample #{}, Class: {}'.format(train, num+1, classification) # adding +1 to the num so that the title doesn't start with 0 (even though we are indexing the 0 position)
        plt.title(str1)
        plt.show()
        
    
# Find the all-zero rows in the EEG samples, calculate the percentage, and remove them 
def find_zero(signals, label, contains_labels):
    binary_findings = (signals == 0).all(axis=2) # produce a binary matrix determining which channels are all zeros (therefore dummy channels) - this removes the dimension with 256 points
    single = binary_findings.any(axis=1) # produces a binary vector of each instance stating whether or not it contains an all-zeros channel (true = contains all zeros, false = does not contain all zeros)
    percentage = np.count_nonzero(single == 1)/len(signals) # portion of the dataset which contains an all-zero channel
    new_signal = signals[~single] # copy of the signal dataset with ONLY NON-ZERO CHANNELS
    if contains_labels == 1:
        new_label = label[~single] # copy of the labels vector with instances that contain ONLY NON-ZERO CHANNELS
    print('Percentage of instances that contain at least 1 null channel: {}%'.format(str(round(percentage, 5)*100)))
    # What is the size and class balance after removing zero entries...
    print('-------- AFTER ZERO ENTRY REMOVAL: --------')
    print('New signal data size: {}'.format(new_signal.shape))
    if contains_labels == 1: # if this dataset includes labels (i.e. training or validation)...
        print('New label vector size: {}'.format(new_label.shape))
        print('Percentage of seizure occurences in cleaned dataset: {}%'.format(str(round((np.count_nonzero(new_label == 1)/len(new_label))*100, 3))))
        return new_signal, new_label
    else: 
        return new_signal
    
    
# Applying continuous wavelet decomposition 
   # signal.cwt adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html
def cwt_convert(signals, label, width_used, percentage, contains_labels):
    cut = math.ceil(len(signals)*percentage) # the number of datapoints we will keep
    # want to shuffle data before cutting it so that we get a different combination each time
    if contains_labels == 1:
        signal_shuffle, label_shuffle = shuffle(signals, label, random_state=27)
        # extract subset of data - since it is randomly shuffled we can scrape the top number of instances
        signals_cut = signal_shuffle[0:cut] # extracting the signal of the instances we are keeping 
        label_cut = label_shuffle[0:cut] # extract the corresponding labels for the instances we are keeping
    else:
        signal_shuffle = shuffle(signals, random_state=27)
        signals_cut = signal_shuffle[0:cut] 
    cwt_whole_list = [] # empty list to hold the four dimensional final data structure
    for i in range(signals_cut.shape[0]): # for each instance.. 
        cwt_data_list = [] # create an empty list to hold each instance information
        for c in range(signals_cut.shape[1]): # for each channel in this instance
            cwtmatr = signal.cwt(signals_cut[i, c, :], signal.ricker, width_used) # extracting cwt transform for a SINGLE channel (new channel shape: 30 x 256)
            cwtmatr_list = list(cwtmatr) # convert to a list to allow  for proper appending
            cwt_data_list.append(cwtmatr_list) # append each channel (now matrix) on top of each other 
        cwt_instance = np.array(cwt_data_list) # create a np array for this entire instance  (1, 23, 30, 356)
        cwt_whole_list.append(cwt_instance) # append the completed instance to the whole dataset
    cwt_data = np.array(cwt_whole_list) # convert the dataset to a numpy array for easy indexing
    if contains_labels == 1: # if this dataset includes labels...
        return cwt_data, label_cut # return both the cwt transformed signals and the new, shortened label vector
    else: # or else...
        return cwt_data # only return the cwt transformed signals
    
    
# Scalogram - adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html
def scalogram_plot(channel_matrix, label):
    plt.figure()
    plt.imshow(channel_matrix, extent=[-1, 1, 31, 1], aspect='auto', # cmap='PRGn',
           vmax=abs(channel_matrix).max(), vmin=-abs(channel_matrix).max()) # change extent depending on widths used
    plt.title('Scalogram Plot Example, Class: {}'.format(class_labels[label]))
    plt.show()



# ----------------------------------------------------------------------------
## Load in each of the .npz files and detail their information 

print('----------------------------------------------------------------------------\n1. Information on data files')

# eeg-seizure_train.npz
name4 = 'eeg-seizure_train.npz'
seiz_train = np.load('data/'+name4, allow_pickle=True) # file type = lib.npyio.NpzFile. 
seiz_train_signals = seiz_train['train_signals']
seiz_train_labels = seiz_train['train_labels']
print_info(name4, seiz_train, seiz_train_signals, seiz_train_labels, 1)
seiz_train_signals1, seiz_train_labels1 = find_zero(seiz_train_signals, seiz_train_labels, 1)

    
# eeg-seizure_val_balanced.npz
name6 = 'eeg-seizure_val_balanced.npz'
seiz_val_bal = np.load('data/'+name6, allow_pickle=True) # file type = lib.npyio.NpzFile. 
seiz_val_bal_signals = seiz_val_bal['val_signals']
seiz_val_bal_labels = seiz_val_bal['val_labels']
print_info(name6, seiz_val_bal, seiz_val_bal_signals, seiz_val_bal_labels, 1)
seiz_val_bal_signals1, seiz_val_bal_labels1 = find_zero(seiz_val_bal_signals, seiz_val_bal_labels, 1)

# eeg-seizure_test.npz
name7 = 'eeg-seizure_test.npz'
seiz_test = np.load('data/'+name7, allow_pickle=True) # file type = lib.npyio.NpzFile. 
seiz_test_signals = seiz_test['test_signals']
print_info(name7, seiz_test, seiz_test_signals, 0, 0)
seiz_test_signals1 = find_zero(seiz_test_signals, 0, 0)



# ----------------------------------------------------------------------------
## Short exploratory analysis of EEG data sample 

num_list = list(range(5, 15))
looking_at_waves = plot_together(seiz_train_signals1, num_list, seiz_train_labels1, 'Training')

# Interested in seeing what the raw data looks like - especially when differentiating between 
# seizure vs non-seizure recordings. 



# ----------------------------------------------------------------------------
## Transforming channel signals to continuous wavelet transforms 

widths = np.arange(1, 4) # number of signals produced by the  CWT
length = len(widths)
# bc i do not have enough memory, i will use a subset of the data (these are the percentages I will use)
train_to_use = 0.1
val_to_use = 0.18
test_to_use = 0.09

# eeg-seizure_train.npz
cwt_seiz_train_signals1, seiz_train_labels1_cut = cwt_convert(seiz_train_signals1, seiz_train_labels1, widths, train_to_use, 1)

# eeg-seizure_val_balanced.npz
cwt_seiz_val_bal_signals1, seiz_val_bal_labels1_cut = cwt_convert(seiz_val_bal_signals1, seiz_val_bal_labels1, widths, val_to_use, 1)

# eeg-seizure_test.npz
cwt_seiz_test_signals1 = cwt_convert(seiz_test_signals1, 0, widths, test_to_use, 0) 



# ----------------------------------------------------------------------------
## Visualizing CWT

# Scalogram example
inst = 2
channel = 12
scalogram_plot(cwt_seiz_train_signals1[inst, channel, :, :], seiz_train_labels1_cut[inst])

# Can also visualize the CWT output using plot...
plt.figure()
for i in range(length):
    plt.plot(cwt_seiz_train_signals1[inst, channel, i, :])
plt.title('Example of CWT Signal, Class: {}'.format(class_labels[seiz_train_labels1_cut[inst]]))


# ----------------------------------------------------------------------------
## Neural network model - adapted from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# reshape data to fit in the NN model correctly 
x_train = cwt_seiz_train_signals1.reshape(cwt_seiz_train_signals1.shape[0], 23, -1) # must conver the data from 4 dimensions to 3, will "flatten" the third and fourth axes to do this
y_train = seiz_train_labels1_cut.reshape(-1, 1)
x_val = cwt_seiz_val_bal_signals1.reshape(cwt_seiz_val_bal_signals1.shape[0], 23, -1) # must conver the data from 4 dimensions to 3, will "flatten" the third and fourth axes to do this
y_val = seiz_val_bal_labels1_cut.reshape(-1, 1)
x_test = cwt_seiz_test_signals1.reshape(cwt_seiz_test_signals1.shape[0], 23, -1) # must conver the data from 4 dimensions to 3, will "flatten" the third and fourth axes to do this

# define model - using sequential with several fully connected (dense layers)
model = Sequential() # initialize the keras model 
model.add(Dense(8, input_shape=(23, length*256,), activation='relu')) # add a dense layer with 8 nodes, relu activation, and the expected input size
model.add(Dense(4, activation='relu')) # next, add another dense layer with 4 nodes and relu activation
model.add(Dense(1, activation='sigmoid')) # finally add one last dense layer, which uses the sigmoid activation function

# compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
fitted_model = model.fit(x_train, y_train, epochs=80, validation_data=(x_val, y_val), batch_size=10)

# evaluate the model with the accuracy 
train_acc = model.evaluate(x_train, y_train) 
test_acc = model.evaluate(x_val, y_val)



# ----------------------------------------------------------------------------
## Visualization of model accuracy 

# Plot training and validation accuracies on a plot 
plt.figure()
plt.title('Model Training & Validation Accuracy')
plt.plot(fitted_model.history['accuracy'], label='Training')
plt.plot(fitted_model.history['val_accuracy'], label='Validation')
plt.legend()
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.show()

# Prediction on test data
# predict_x = model.predict(x_test) #<< having an issue
# y_pred = np.argmax(predict_x, axis=1) #<< having an issue









