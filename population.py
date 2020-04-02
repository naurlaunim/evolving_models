import copy

from individual import ModelIndividual


class Population:
    def __init__(self, pop_size, max_epochs):
        self.pop_size = pop_size
        self.models = [ModelIndividual(max_epochs) for i in range(pop_size)]
        self.num_gen = 0
        self.data = self.prepare_data()

    def evaluate(self, verbose):
        X_train, y_train, X_test, y_test = self.data

        for model in self.models:
            if verbose:
                print(model.units_in_layers)
            model.evaluate(X_train, y_train, X_test, y_test, verbose)

    def mutate(self):
        new_pop = [copy.deepcopy(ind) for ind in self.models]
        for ind in new_pop:
            ind.mutate()
        return new_pop

    def generation(self, verbose):
        new_pop = self.mutate()
        self.models += new_pop
        self.evaluate(verbose)
        self.models.sort(key=lambda model: model.fitness, reverse=True)
        self.models = self.models[:self.pop_size]

    def __str__(self):
        l = [str(model) for model in self.models]
        return str(self.num_gen) + ': ' + ', '.join(l)

    def prepare_data(self):
        from keras.datasets import mnist
        # download mnist data and split into train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # reshape data to fit model
        X_train = X_train.reshape(60000, 28, 28, 1)
        X_test = X_test.reshape(10000, 28, 28, 1)

        from keras.utils import to_categorical
        # one-hot encode target column
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        return X_train, y_train, X_test, y_test

    def evolve(self, gen, verbose):
        print(self)
        for i in range(gen):
            self.generation(verbose)
            self.num_gen += 1
            print(self)
