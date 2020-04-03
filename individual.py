import random
import math

from create_model import create_model


class ModelIndividual:
    def __init__(self, max_epochs):
        self.max_f = 7 # filters = 2^max_f
        self.max_k = 4 # kernel size = 2*max_k + 1
        self.max_l = 5 # layers < max_l
        self.layers_number = random.randrange(2, self.max_l)
        self.units_in_layers = [{'filters': 2**random.randrange(2, self.max_f), 'kernel': 1 + 2*random.randrange(1, self.max_k)} for i in range(self.layers_number)]
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.max_epochs = max_epochs
        self.max_param = (1 + 2*self.max_k)*2 * 2**self.max_f * 1 + 2**self.max_f + ((1 + 2*self.max_k)*2 * 2**self.max_f * 2**self.max_f + 2**self.max_f) * (self.max_l - 2)
        self.fitness = None
        self.accuracy = None
        self.trained_on_epochs = None
        self.finesse = None
        self.compute_finesse()

    def compute_finesse(self):
        self.finesse = 1 - math.log(self.count_hidden_parameters()) / math.log(self.max_param)


    def evaluate(self, X_train, y_train, X_test, y_test, verbose):
        input_shape = X_train[0].shape
        output_shape = y_train[0].shape[0]
        model = create_model(input_shape, output_shape, self.layers_number, self.units_in_layers)
        model.compile(self.optimizer, self.loss, self.metrics)
        self.trained_on_epochs = self.max_epochs
        for epoch in range(self.max_epochs):
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, verbose=verbose)
            train_acc, val_acc = history.history.get('accuracy')[-1], history.history.get('val_accuracy')[-1]
        self.compute_fitness(val_acc)
        return self.fitness

    def mutate(self):
        # with low probability add or remove layer
        add_layer =  random.random() < 0.05
        remove_layer = self.layers_number > 1 and random.random() < 0.05

        if add_layer:
            self.add_layer()
        if remove_layer:
            self.remove_layer()

        # with high probability add or remove unit in layer
        if not (add_layer or remove_layer):
            layer = random.randrange(self.layers_number)
            if random.random() < 0.5:
                self.change_filters_number(layer)
            else:
                self.change_kernel(layer)

    def add_layer(self):
        unit = {'filters': 2**random.randrange(2, self.max_f), 'kernel': 1 + 2*random.randrange(1, self.max_k)}
        self.units_in_layers.insert(random.randrange(self.layers_number), unit)
        self.layers_number += 1

    def remove_layer(self):
        del self.units_in_layers[random.randrange(self.layers_number)]
        self.layers_number -= 1

    def change_filters_number(self, layer):
        filters = self.units_in_layers[layer].get('filters')
        steps = [-2**i for i in range(3, 0, -1)] + [2**i for i in range(1, 4)]
        steps = [step for step in steps if filters+step >= 2]
        self.units_in_layers[layer]['filters'] = filters + random.choice(steps)

    def change_kernel(self, layer):
        kernel = self.units_in_layers[layer].get('kernel')
        step = random.choice([-2, 2])
        if kernel + step >= 1:
            self.units_in_layers[layer]['kernel'] = kernel + step

    def compute_fitness(self, accuracy):
        self.accuracy = accuracy
        self.compute_finesse()
        self.fitness =  2/(1/accuracy + 1/self.finesse)

    def __str__(self):
        layers = [[(l.get('kernel'), l.get('kernel')),l.get('filters')] for l in self.units_in_layers]
        return '('+str(layers) + ', fin: '+ str(round(self.finesse, 4)) + ', ' + (('acc: '+ str(round(self.accuracy, 4)) + ', fit: '+ str(round(self.fitness, 4)) + ', ep: ' + str(self.trained_on_epochs)) if self.fitness else 'newborn')+')'

    def count_hidden_parameters(self):
        conv = self.units_in_layers[0]
        count = conv.get('kernel') * 2 * conv.get('filters') * 1 #MNIST 28x28x1
        count += conv.get('filters')  # + bias
        for i in range(1, self.layers_number):
            conv = self.units_in_layers[i]
            count += conv.get('kernel')*2*conv.get('filters')*self.units_in_layers[i-1].get('filters') + conv.get('filters') # + bias
        return count
