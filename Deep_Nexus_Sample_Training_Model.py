import tensorflow as tf
from keras import backend as K

###################################
# On GPU systems these parameters may be required:
''' 
TensorFlow wizardry
Due to GPU RAM bottlenecks and the tendency for GPUs to blow thru and max out RAM, 
it is necessary to configure and allocate GPU memory with the following code:
'''
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of n % of the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.98

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
###################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.metrics import explained_variance_score
from keras.models import *
from keras.models import load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import backend as K
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras import backend
import time

# Huber loss function for training
def smooth_L1_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)

# Keras with Tensorflow backend; x-argument is variable or tensor; returns a tensor
# activation function of non-linear transformations & learning complex tasks;back-propagation, updating weights & biases
# operates on one element of the tensor at a time, performing exponential calculations
def custom_activation_exp(x):
    return backend.exp(x)

# Keras with Tensorflow backend; x-argument is variable or tensor; returns a tensor
# activation function of non-linear transformations & learning complex tasks;back-propagation, updating weights & biases
# operates on one element of the tensor at a time, performing logarithmic calculations
def custom_activation_log(x):
    return backend.log(x)

# custom loss function to account for direction of prediction (whether positive or negative change)
# penalizes model for predicting wrong direction of prediction (where MAE won't bc of absolute values)
# using this or not?
def stock_loss(y_true, y_pred):
    alpha = 100.
    loss = K.switch(K.less(y_true * y_pred, 0), \
        alpha*y_pred**2 - K.sign(y_true)*y_pred + K.abs(y_true), \
        K.abs(y_true - y_pred)
        )
    return K.mean(loss, axis=-1)

# quantile regression loss function
# good for estimating certainty over a dataset distribution as opposed to a single datapoint
# using this or not?
def q_loss(q, y, f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1) # axis -1 is standard
quantile = 0.50 # median (50th percentile)

# measures difference between actual/observed value and predicted value
# measure of accuracy, comparing forecasting errors of different models for a particular dataset
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# this can be used as a metric. 1 is a perfect score; an exact match to the target
# proportion of variance in dependent variable that is predictable by independent variable(s)
# provides a measure of how well observed/actual outcomes are replicated by the model
def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def SuperSmooth(ssdata, period=50): # 50 is default period
    '''
    SuperSmooth - proposed by John Ehler to get rid of undersample noise.
    Digital signal processing
    The idea is to smooth the data (eliminate as much noise as possible) with as little lag as possible.
    Combination of 2-pole Butterworth filter and 2-bar SMA
    Increasing number of poles increases the lag
    '''
    # close = df[['close']].values.ravel()
    f = (1.414 * np.pi) / period
    a = np.exp(-f)
    c2 = 2 * a * np.cos(f)
    c3 = - a * a
    c1 = 1 - c2 - c3

    ss = np.zeros(ssdata.shape) # returns empty array of zeros in shape of ssdata dataset
    for i in range(len(ssdata)):
        if i < 2:
            continue
        else:
            ss[i] = c1 * (ssdata[i] + ssdata[i - 1]) * 0.5 + c2 * ss[i - 1] + c3 * ss[i - 2]
    # df_ss = pd.DataFrame(data=ss, index=df.index, columns=['ss_{i}'.format(i=period)])
    return ss

np.random.seed(1335)  # for reproducibility


# read in csv file of desired dataset to Pandas
# must specify the index column so that the dataframe can work with the timestamps
df = pd.read_csv('.csv', index_col='Date')

# ways of splitting up large dataframe into smaller dataframes

#a, ticker = np.array_split(df, 2)
a, b, ticker = np.array_split(df, 3) # using to split dataframe into 3 parts b/c dataset is so large

print(ticker.shape) # confirm rows, columns, and dimensions of dataset
print(ticker.head()) # prints first 5 lines of dataset
print(ticker.tail()) # prints last 5 lines of dataset


timeindex = pd.to_datetime(ticker.index) # converting index to datetime format

t_size = .20 # test-size, in this case 20% of set while 80% is train-size in data set split in sk-learn


start_time = time.time() # use the time module to measure how much (wall clock) time elapses during processing
print('begin signal processing...')

ssdata0 = ticker.Close
ssdata = np.array(ssdata0) # must convert to numpy array for ss to run calculations

# creating column in the ticker dataframe of signal processed Close
ticker['ss'] = SuperSmooth(ssdata, period=60)  # defining period window


print('signal processing complete.') # data has been signal processed
print("time elapsed: {:.2f}s".format(time.time() - start_time))


# performing log returns (base 2) on pre-processed ticker data
# this is done with 3 time periods: 180 minutes, 90 minutes, 1 minute
# looking at price action in multiple dimensions
# enables the neural network to understand context in the price action
# must have measurements of change, as raw price will not work
# max measurement of change over time cannot exceed sequence length
ticker['target'] = np.log2(ticker.ss / ticker.ss.shift(180)).fillna(0.0001)
#ticker['target0'] = np.log2(ticker.ss2 / ticker.ss2.shift(90)).fillna(0.0001)
ticker['target1'] = np.log2(ticker.ss / ticker.ss.shift(90)).fillna(0.0001)
ticker['target2'] = np.log2(ticker.ss / ticker.ss.shift(1)).fillna(0.0001)


# creating arrays of the log-returns data
targetnp = np.array(ticker.target)
#targetnp0 = np.array(ticker.target0)
target1np = np.array(ticker.target1)
target2np = np.array(ticker.target2)


# stacking all log-returns arrays into one large 2D array to run thru neural network
data = np.column_stack((targetnp, target1np, target2np))


Xdata0 = np.delete(data, np.s_[0:500], axis=0)  # slice off initial nan's from array

# early scaling, can possibly be moved to after train/test split.
print('scaling...')
scaler = preprocessing.RobustScaler()  # removed for 1d data
Xdata = scaler.fit_transform(Xdata0)

X = np.array(Xdata[:, :]) #selects the range of columns up to the number passed ??
#X = np.array(Xdata[:, 0])
#Y = np.array(Xdata[:, 4]) # target
Y = np.array(Xdata[:, 0])  # target
#Y = sg.savgol_filter(Y0, 11, 1, axis=-0)

#Y = np.array(close_1[:, 3])
print(' ')
print('X and Y Shape')
print(X.shape)
print(Y.shape)

# name variables for arrays of sequences to be created
dX, dY = [], []

n_pre = 180 # input sequence length (n actual/observed timesteps)
n_post = 15 # output sequence length (n predictions)

# for loop to create 3D arrays to pass thru LSTM layers, specifically RepeatVector and TimeDistributed
for i in range(len(X) - n_pre - n_post):
    dX.append(X[i:i + n_pre])
    #dX.append(sg.savgol_filter(X[i:i + n_pre], 21, 5, axis=-0))
    dY.append(Y[i + n_pre:i + n_pre + n_post])
    #dY.append(sg.savgol_filter(Y[i + n_pre:i + n_pre + n_post], 21, 3, axis=-0))
dataX = np.array(dX)
####dataX = np.reshape(dataX1, (-1, n_pre, 1))  #in case of single input/feature
dataY1 = np.array(dY)
dataY = np.reshape(dataY1, (-1, n_post, 1)) # create 3D array from one column of data

#np.savetxt('.csv', dataX, delimiter=',')
#np.savetxt('.csv', dataY1, delimiter=',')


print(' ')
print('dataX...')
print(dataX)
print(' ')
print('dataY...')
print(dataY)
print(dataX.shape)
print(dataY.shape)

# configure LSTM
# name variables for training and testing arrays to run thru LSTM
# this is from sk-learn, splitting up dataset into training and testing segments, no shuffling before splitting
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=t_size, shuffle=False)

epochs = 100 # number of times the LSTM model will pass thru entire dataset
batch_size = 512 # number of samples in dataset processed by LSTM before model is updated

# must use an optimizer with Keras model
# Adam optimizer for stochastic gradient descent
# lr is learning rate or step size
# __________ more here
# experimentation necessary
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# create LSTM (remember LSTM requires 3D array)
model = Sequential() # neural network of linear stack of layers
# add layers as needed below:

# bidirectional allows outputs to get info from forward (future) and backward (past) states simultaneously
# input_shape is dimensions of array to be trained; return_sequences shows hidden state outputs if set to True
# CuDNN enables GPU processing with CUDA
model.add(Bidirectional(CuDNNLSTM(512, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)))

# relu accounts for interactions and non-linearities between inputs
model.add(Activation('relu'))

# repeat input n number of times -- in this case n_post -- which is number of prediction outputs or output sequence
model.add(RepeatVector(n_post))

# CuDNN enables GPU processing with CUDA
# in this layer must show hidden state outputs as sequence before adding timedistributed layer
model.add(CuDNNLSTM(512, return_sequences=True))

model.add(Activation('relu'))
#model.add(RepeatVector(12))
#model.add(LSTM(256, return_sequences=True))
#model.add(Activation('relu'))

# time distributed layer a must for sequence-to-sequence prediction, performing all operations at each timestep (Dense)
# use of time is embedded in the neural network architecture, rather than as a feature
model.add(TimeDistributed(Dense(1)))

model.add(Activation('linear')) # defines final range of numbers being output, basically returning itself in this model

# configure the learning process before training the model
# arguments include: loss function, optimizer, and custom metrics as defined above
model.compile(loss=smooth_L1_loss, optimizer=optimizer, metrics=[coeff_determination])

# gives view of internal state and statistics of model as it is training
# save model after every epoch, saving the best version (weights, biases, etc)
# verbosity (0 or 1) : whether or not to log messages of improvements while model is training
checkpointer = ModelCheckpoint(filepath=".hdf5", save_weights_only=False, verbose=1, save_best_only=True)

# trains model for given number of epochs (as specified above) with x-array input training data,
# y-array output target data, validation data is set aside to evaluate loss and metrics at end of each epoch (model
# is NOT trained on this portion of the dataset)
# batch size (as specified above), early stopping means to stop training once model stops
# improving on validation loss by a minimum amount (min_delta), patience is how many epochs to let pass before
# determining no further improvement, verbosity logs messages of improvements
# shuffle is a Boolean; if True, data is randomly shuffled before each epoch
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=batch_size, epochs=epochs,
                    callbacks=[checkpointer, EarlyStopping(monitor='val_loss',
                                                           min_delta=5e-5, patience=20, verbose=1)],
                    shuffle=True)

# use load_model to accommodate adding of custom layers (termed custom_objects below)
# explicitly instruct to load the BEST model saved, not just the training epoch
model = load_model('.hdf5', custom_objects={'smooth_L1_loss': smooth_L1_loss, 'coeff_determination': coeff_determination})

# summary representation of the model
print(model.summary())

# run predictions based on BEST model saved (could be epochs ago)
# predictions based on input array (X_test), verbosity for logging messages
# batch_size for predict as large as possible to speed up output (default batch_size=32 takes FOREVER)
predict = model.predict(X_test, batch_size=768, verbose=1)
print('')
print('Predictions & Shape')
print(predict.shape)
print(predict)


# now plot
nan_array = np.empty((n_pre - 1))
nan_array.fill(np.nan)
nan_array2 = np.empty(n_post)
nan_array2.fill(np.nan)
ind = np.arange(n_pre + n_post)

fig, ax = plt.subplots()
for i in range(0, 100, 100):
    forecasts = np.concatenate((nan_array, X_test[i, -1:, 0], predict[i, :, 0]))
    ground_truth = np.concatenate((nan_array, X_test[i, -1:, 0], y_test[i, :, 0]))
    network_input = np.concatenate((X_test[i, :, 0], nan_array2))
    print('')
    print('Forecasts')
    print(forecasts.shape)
    print(forecasts)

    ax.plot(ind, network_input, 'b-x', label='Network input')
    ax.plot(ind, forecasts, 'r-x', label='Many to many model forecast')
    ax.plot(ind, ground_truth, 'g-x', label='Ground truth')

    plt.xlabel('t')
    plt.ylabel('sin(t)')
    plt.title('Sinus Many to Many Forecast')
    plt.legend(loc='best')
    plt.savefig('goldD Final 1hr UTC' + str(i) + '.png')
    plt.cla()

# convert to 2D. 12 (or the final number in reshape) is number of predictions per row.
# must convert to 2D in order to save in CSV for evaluation and analysis
predict_flat = predict.transpose(2, 0, 1).reshape(-1, n_post)
y_flat = y_test.transpose(2, 0, 1).reshape(-1, n_post)

# for analysis -- calculating variance score below
predict_flat1 = predict.reshape(-1, 1)
y_flat1 = y_test.reshape(-1, 1)


# save predictions here
np.savetxt('.csv', predict_flat, delimiter=',')
np.savetxt('.csv', y_flat, delimiter=',')

# variance regression score function in sk_learn
# how close are predicted values to actual values
# y_flat1 is ground truth, predict_flat1 is predictions, multioutput uniform_average takes an average score of all
# outputs with uniform weight applied to them
var = explained_variance_score(y_flat1, predict_flat1, multioutput='uniform_average')
print('')
print('Variance Score: 1.00 = perfect match')
print(var)



