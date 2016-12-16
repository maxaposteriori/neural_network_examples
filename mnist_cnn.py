from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, \
                         MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# Load the data.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten images.
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# Normalize.
X_train = X_train / 255
X_test = X_test / 255

# One-hot encode output.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the model.
model = Sequential([
	Convolution2D(
        30, 5, 5, border_mode='valid', input_shape=(1, 28, 28),
        dim_ordering="th"),
	Activation('relu'),
	MaxPooling2D(pool_size=(2, 2)),
	Convolution2D(15, 3, 3),
	Activation('relu'),
	MaxPooling2D(pool_size=(2, 2)),
	Dropout(0.2),
	Flatten(),
	Dense(128),
	Activation('relu'),
	Dense(50),
	Activation('relu'),
	Dense(10),
	Activation('softmax')])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Fit the model.
model.fit(X_train, y_train, validation_split=0.1)

# Test the model.
res = model.evaluate(X_test, y_test)
print("\nEvaluation...")
print("   Accuracy: {acc}".format(acc=res[1]))

