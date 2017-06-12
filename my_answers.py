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
        
    # reshape each asarray
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y




# TODO build the required RNN model: a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
#starts sequential model
model = Sequential()
# # LSTM layers of 200 and input shapes of window size 
model.add(LSTM(200, input_shape=(window_size,len(chars))))
# Dense layer using linear activation function
model.add(Dense(len(chars), activation='linear'))
# Activation layer using softmax activation funtion
model.add(Activation('softmax'))
# prints layers,output and any Parameters
model.summary()

# initialize optimizer
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile model --> make sure initialized optimizer and callbacks - as defined above - are used
model.compile(loss='categorical_crossentropy', optimizer=optimizer)






### TODO: list all unique characters in the text and remove any non-english ones
# find all unique characters in the text

#First duplicate the text to test_text to find all the non-english characters in the text

test_text = text
valid_english_characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '!', ',', '.', ':', ';', '?']
for i in range(0,len(valid_english_characters)):
    test_text = test_text.replace(valid_english_characters[i],'')
#List all non-valid english characters(special character).
nonvalid_english_characters = list(set(test_text))
#Print all non-valid english characters 
print(nonvalid_english_characters)
#Remove non-valid english characters (special Characters) from text 
nonvalid_english_characters = ['/', '(', '"', '&', '%', '*', ')', '-', '$', '@', "'", '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'è', 'é',  'à', 'â']
for i in range(0,len(nonvalid_english_characters)):
    text = text.replace(nonvalid_english_characters[i],'')
    
# shorten any extra dead space created above by replaceing special characters
text = text.replace('  ',' ')




### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    # loop for that goes through inputed texts and outputed text until end
    inputs = [text[i:i+window_size] for i in range(0,len(text)-window_size)]
    outputs = [text[i+window_size] for i in range(0,len(text)-window_size)]
    

    
    return inputs,outputs
