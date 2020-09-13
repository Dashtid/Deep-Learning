from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D


def twoLayerMLP(n_unit):
    model = Sequential()
    model.add(Dense(n_unit, input_dim=2, activation='relu'))  # Creating first layer with activation function 'Relu'
    model.add(Dense(1, activation='sigmoid'))  # Creating a second layer with activation function 'Sigmoid'

    # Compiling the layers and setting loss & optimizer functions
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(learning_rate=0.1),
                  metrics=['binary_accuracy'])

    # Executing training with 2000 epochs
    model.fit(Input, Target, epochs=2000, verbose=0)

    # Printing results
    print("The predicted class labels are:", model.predict(Input))
