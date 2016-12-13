from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# Load the data.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten images.
X_train = X_train.reshape((-1, X_train.shape[1] * X_train.shape[2]))
X_test = X_test.reshape((-1, X_test.shape[1] * X_test.shape[2]))

# Normalize.
X_train = X_train / 255
X_test = X_test / 255

# One-hot encode output.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the model.
model = Sequential([
    Dense(10, input_shape=(X_train.shape[1],)),
    Activation('softmax')])
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Fit the model.
model.fit(X_train, y_train, validation_split=0.1)

# Test the model.
res = model.evaluate(X_test, y_test)
print("\nEvaluation...")
print("   Accuracy: {acc}".format(acc=res[1]))

