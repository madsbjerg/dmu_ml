import string
import random
import numpy.random as npr
import numpy as np
#Solution
solution = "To be or not to be that is the question"

#Fitness function (opgave A)
def fitness(string):
    fitness = 0

    for i in range(len(string)):
        if string[i] == solution[i]:
            fitness += 1

    return fitness

#Random strings (opgave B)
highest_fitness = 0
highest_fitness_string = ''
attempts = 1000

for i in range(attempts):
    letters = string.ascii_letters + ' '
    random_str = ''.join(random.choice(letters) for j in range(len(solution)))

    if fitness(random_str) > highest_fitness:
        highest_fitness = fitness(random_str)
        highest_fitness_string = random_str

print('Random strings result after ' + str(attempts) + ' attempts')
print('Highest fitness: ' + str(highest_fitness) + ' was ' + highest_fitness_string)

# Hillclimbing (Opgave 3)

# Set up initial state (a random string)
print('Starting hillclimb')

letters = string.ascii_letters + ' '
initial_state = ''.join(random.choice(letters) for j in range(len(solution)))
print('Initial string: ' + initial_state + ' with fitness ' + str(fitness(initial_state)))

# Implement function to generate neighbours to this string
def generate_neighbours(input_str, amount):
    neighbours = []

    for i in range(amount):
        new_str = ''
        for j in range(len(input_str)):
            if not input_str[j] == solution[j]:
                new_str += random.choice(letters)
            else:
                new_str += input_str[j]
        #print('Adding to neighbours: ' + new_str)
        neighbours.append(new_str)

    return neighbours

def hillclimb(initial, iterations, amount_of_neighbours):
    current_string = initial
    string_fitness = 0
    current_iteration = 0

    for i in range(int(iterations)):
        neighbours = generate_neighbours(current_string, amount_of_neighbours)
        for j in range(len(neighbours)):
            if fitness(neighbours[j]) > string_fitness:
                string_fitness = fitness(neighbours[j])
                current_string = neighbours[j]
        current_iteration += 1

    return [current_string, string_fitness, current_iteration]


results = hillclimb(initial_state, 90, 5)
print('Finished in iterations: ' + str(results[2]))
print('Best result: ' + results[0] + ' with fitness: ' + str(results[1]))

# Genetic Algorithm (Opgave D)
import string
import random
import numpy.random as npr

letters = string.ascii_letters + ' '
solution = "To be or not to be that is the question"
#solution = "To be"

#A possible solution is a chromosome, each character is a gene
#Could also be called "Individual"
class Chromosome:
    genes = []
    fitness = 0
    def __init__(self):
        self.genes = [random.choice(letters) for char in solution]

    def gene_string(self):
        g_string = ''
        for gene in self.genes:
            g_string += gene
        return g_string
    def calculate_fitness(self):
        fit = 0
        for i in range(len(self.genes)):
            if self.genes[i] == solution[i]:
                fit += 1

        self.fitness = fit
        return fit


#Population is the population of chromosomes or individuals
class Population:
    pop_size: int
    chromosomes = []
    fittest: int

    def __init__(self, population_size: int):
        self.initialize_population(population_size)

    def initialize_population(self, population_size: int):
        self.chromosomes = [Chromosome() for i in range(population_size)]
        self.pop_size = population_size

    def get_fittest(self):
        max_fit = -1
        max_fit_index = -1
        for i in range(len(self.chromosomes)):
            if max_fit < self.chromosomes[i].calculate_fitness():
                max_fit = self.chromosomes[i].fitness
                max_fit_index = i

        self.fittest = self.chromosomes[max_fit_index].fitness
        return self.chromosomes[max_fit_index]


# Class for doing selection, crossover and mutation
class MonkeysGA:
    population: Population
    population_size: int
    generation_count: int
    fittest: Chromosome
    mutation_rate: int #1..100 based on how likely you want genes to mutate
    crossover_type: int #1..3 based on what type of crossover is selected, e.g random or uniform
    mating_pool = []

    def __init__(self, pop_size: int, mutation_rate: int, crossover_type: int):
        self.population = Population(pop_size)
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_type = crossover_type
        self.generation_count = 0

    def selection(self):
        # using roulette wheel probability
        self.mating_pool = []
        # Gets the sum fitness of our population
        population_fitness = sum([chromosome.fitness for chromosome in self.population.chromosomes])

        # Calculate the probabilities of each chromosome in the pop
        chromosome_probabilities = [chromosome.fitness / population_fitness
                                    for chromosome in self.population.chromosomes]

        # use numpy random to select an amount of possible mates based on population size
        amount = self.population_size / 10
        self.mating_pool = npr.choice(self.population.chromosomes, int(amount), p=chromosome_probabilities)

    # After this function is called first_fittest and second_fittest are now offspring and no longer parents
    # This may also be called our "breeding" function
    def crossover(self, parent1: Chromosome, parent2: Chromosome):
        child = Chromosome()
        match self.crossover_type:
            # Random crossover
            case 1:
                crossover_point = random.randrange(0, len(solution))
                for i in range(len(parent1.genes)):
                    if i < crossover_point:
                        child.genes[i] = parent1.genes[i]
                    else:
                        child.genes[i] = parent2.genes[i]
                return child
            # Uniform crossover, weighted for fittest parent
            case 2:
                crossover = random.randrange(0,9)
                fittest_parent = parent1 if parent1.calculate_fitness() > parent2.calculate_fitness() else parent2
                least_fit_parent = parent1 if parent1.calculate_fitness() < parent2.calculate_fitness() else parent2
                for i in range(len(parent1.genes)):
                    if crossover < 7:
                        child.genes[i] = fittest_parent.genes[i]
                    else:
                        child.genes[i] = least_fit_parent.genes[i]
                return child
            # Uniform crossover 50/50 weighting
            case 3:
                crossover = random.randrange(1, 10)
                for i in range(len(parent1.genes)):
                    if crossover <= 5:
                        child.genes[i] = parent1.genes[i]
                    else:
                        child.genes[i] = parent2.genes[i]
                return child

    def mutation(self, child: Chromosome):
        index = random.randrange(0,len(child.genes))
        if random.randrange(0,100) <= self.mutation_rate:
            child.genes[index] = random.choice(letters)
        return child

    def run(self):
        print('Running GA')
        # Calculate fitness for all chromosomes
        for chrom in self.population.chromosomes:
            chrom.calculate_fitness()

        while self.population.get_fittest().calculate_fitness() < 39:
            self.selection()

            self.fittest = self.population.get_fittest()
            print('Fittest chromosome for generation ' + str(
                self.generation_count) + ': ' + self.fittest.gene_string() + ' with fitness: '
                  + str(self.fittest.calculate_fitness()))

            # Make new generation, replacing about half the old generation
            for i in range(int(self.population_size / 2)):
                parents = random.choices(self.mating_pool, k=2) # grab 2 parents at random from the matingpool
                child = self.crossover(parents[0], parents[1])
                #self.population.chromosomes[i] = self.mutation(child)
                if random.randrange(0,9) < 5:
                    self.population.chromosomes[i] = self.mutation(child)
                else:
                    self.population.chromosomes[i+1] = self.mutation(child)

            self.generation_count += 1

        self.fittest = self.population.get_fittest()
        print('Solution found in gen ' + str(self.generation_count))
        print('Fitness: ' + str(self.fittest.calculate_fitness()))
        print('Gene string: ' + self.fittest.gene_string())


#10000 population, 10 mutation_rate/percentage, random crossover
genetic_algorithm = MonkeysGA(10000, 1, 1)
genetic_algorithm.run()
