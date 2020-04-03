import random

from create_model import create_model


class ModelIndividual:
    def __init__(self, max_epochs):
        self.layers_number = random.randrange(2, 6)
        self.units_in_layers = [random.randrange(4, 20) for i in range(self.layers_number)]
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.max_epochs = max_epochs
        self.max_param = 20**2*5
        self.fitness = None
        self.finesse = 1 - self.count_hidden_parameters() / self.max_param
        self.accuracy = None
        self.trained_on_epochs = None

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
            self.units_in_layers[random.randrange(self.layers_number)] += random.choice([-1, 1])

    def add_layer(self):
        units = random.randrange(20)
        self.units_in_layers.insert(random.randrange(self.layers_number), units)
        self.layers_number += 1

    def remove_layer(self):
        del self.units_in_layers[random.randrange(self.layers_number)]
        self.layers_number -= 1

    def compute_fitness(self, accuracy):
        self.accuracy = accuracy
        count = self.count_hidden_parameters()
        self.finesse = 1 - count/self.max_param
        self.fitness =  2/(1/accuracy + 1/self.finesse)

    def __str__(self):
        return '('+str(self.units_in_layers) + ', fin: '+ str(round(self.finesse, 4)) + ', ' + (('acc: '+ str(round(self.accuracy, 4)) + ', fit: '+ str(round(self.fitness, 4)) + ', ep: ' + str(self.trained_on_epochs)) if self.fitness else 'newborn')+')'

    def count_hidden_parameters(self):
        count = 0
        for i in range(1, self.layers_number):
            count += self.units_in_layers[i-1]*self.units_in_layers[i]
        return count

