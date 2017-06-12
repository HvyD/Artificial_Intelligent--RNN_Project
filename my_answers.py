import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras




# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = [series[i:(i+window_size)] for i in range(len(series)-window_size)]
    y = [series[i+window_size] for i in range(len(series)-window_size)]
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y




# TODO: build an RNN to perform regression on our time series input/output data

model = Sequential()
step_size = 1
# LSTM with 5 hidden layers 
model.add(LSTM(5, input_shape = (window_size,step_size)))
# Dense layer with 1 output node and linear activation function
model.add(Dense(1, activation='linear'))


# model optimizer with keras 
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)

model.summary()




### TODO: list all unique characters in the text and remove any non-english ones
# find all unique characters in the text

#First duplicate the text to test_text to find all the non-english characters in the text

test_text = text
valid_english_characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '!', ',', '.', ':', ';', '?']
for i in range(0,len(valid_english_characters)):
    test_text = test_text.replace(valid_english_characters[i],'')
#List all non-valid english characters.
nonvalid_english_characters = list(set(test_text))
#Print all non-valid english characters 
print(nonvalid_english_characters)
#final non-valid english characters
nonvalid_english_characters = ['/', '(', '"', '&', '%', '*', ')', '-', '$', '@', "'", '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'è', 'é',  'à', 'â']
#Remove non-valid english characters from text 
for i in range(0,len(nonvalid_english_characters)):
    text = text.replace(nonvalid_english_characters[i],'')
    
# shorten any extra dead space created above
text = text.replace('  ',' ')




### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    # Last element of the input/output pair
    rnnElement = window_size
    # Loop until the lastElement of the input/output pair is the last element of the text
    while(rnnElement<len(text)):
        inputs.append(text[rnnElement-window_size:rnnElement])
        outputs.append(text[rnnElement])
        rnnElement += step_size

    
    return inputs,outputs
