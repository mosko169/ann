# Import
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Stock Prediction ANN Parameters
DATA_CSV_PATH = r'Stock.csv'		# Path to a csv containing the historical stock data
STOCK_PARAMETER = 'Open'			# The stock parameter to analyse and predict (Open\High\Low\Close) 
SAMPLE_SIZE = 30					# The amount of consecutive samples in each input to the network (e.g last 30 days)
PREDICTION_OFFSET = 10				# The offset for the network's prediction (e.g the stock's price in 10 days)
TRAINING_SET_PERCENTAGE = 0.8		# The percentage of data to be used for training the network (The rest for testing)
BATCH_SIZE = 16						# The size of each batch of inputs that are inserted into the network
EPOCHS = 10							# The amount of epochs. In each epoch, the data is inserted into the network in multiple batches

# Import data: Take only the needed stock parameter from the csv
data = pd.read_csv(DATA_CSV_PATH)[STOCK_PARAMETER].values

# Create the samples from the raw data
samples = []
for i in range(SAMPLE_SIZE, len(data) - PREDICTION_OFFSET):
	# Each sample contains the values from SAMPLE_SIZE consecutive days, and the stock value in PREDICTION_OFFSET days
	samples.append(np.append(data[i-SAMPLE_SIZE:i], data[i+PREDICTION_OFFSET]))
samples = np.array(samples)

# Divide the samples into training samples and test samples
train_end = int(np.floor(TRAINING_SET_PERCENTAGE*samples.shape[0]))
data_train = samples[:train_end, :]
data_test = samples[train_end+1:, :]

# Build inputs and outputs
train_input = data_train[:, :-1]
train_output = data_train[:, -1]
test_input = data_test[:, :-1]
test_output = data_test[:, -1]

# Amount of neurons in each of the four levels
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

# Session
net = tf.Session()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, SAMPLE_SIZE])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=1)
bias_initializer = tf.zeros_initializer()

# Hidden weights
W_hidden_1 = tf.Variable(weight_initializer([SAMPLE_SIZE, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(test_output)
line2, = ax1.plot(test_output)
plt.show()

# Run
for e in range(EPOCHS):
	# Shuffle training data
	shuffle_indices = np.random.permutation(np.arange(len(train_output)))
	train_input = train_input[shuffle_indices]
	train_output = train_output[shuffle_indices]

	# Run each batch
	for i in range(0, len(train_output) // BATCH_SIZE):
		start = i * BATCH_SIZE
		batch_x = train_input[start:start + BATCH_SIZE]
		batch_y = train_output[start:start + BATCH_SIZE]
		# Run optimizer with batch
		net.run(opt, feed_dict={X: batch_x, Y: batch_y})

	# Show progress
	# MSE train and test
	mse_train = net.run(mse, feed_dict={X: train_input, Y: train_output})
	mse_test = net.run(mse, feed_dict={X: test_input, Y: test_output})
	print('MSE Train: ', mse_train)
	print('MSE Test: ', mse_test)
	
	# Prediction
	pred = net.run(out, feed_dict={X: test_input})
	line2.set_ydata(pred)
	plt.title('Epoch ' + str(e))
	plt.pause(0.01)
