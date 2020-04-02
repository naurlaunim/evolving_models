from keras.models import Sequential
from keras.layers import Dense, Flatten


def create_model(input_shape, output_shape, layers_number, units_in_layers):
    model = Sequential()
    model.add(Dense(units_in_layers[0], activation='relu', input_shape=input_shape))
    for i in range(1, layers_number):
        model.add(Dense(units_in_layers[i], activation='relu'))

    model.add(Flatten())
    model.add(Dense(output_shape, activation='softmax'))
    return model