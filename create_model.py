from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D


def create_model(input_shape, output_shape, layers_number, units_in_layers):
    model = Sequential()
    model.add(Conv2D(units_in_layers[0].get('filters'), units_in_layers[0].get('kernel'), activation='relu', padding='same', input_shape=input_shape))
    for i in range(1, layers_number):
        model.add(Conv2D(units_in_layers[i].get('filters'), units_in_layers[i].get('kernel'), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(output_shape, activation='softmax'))
    return model