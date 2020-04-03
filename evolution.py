from population import Population

pop = Population(pop_size=3, max_epochs=1)
pop.evolve(gen=3, verbose=2)