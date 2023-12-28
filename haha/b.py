import random

class Individual:
    def __init__(self, graph, num_colors):
        self.graph = graph
        self.num_colors = num_colors
        self.chromosome = [random.randint(0, num_colors - 1) for _ in range(len(graph))]

    def fitness(individual):
        # Calculate conflicts
        conflicts = 0
        for i in range(len(individual.graph)):
            for j in individual.graph[i]:
                if individual.chromosome[i] == individual.chromosome[j]:
                    conflicts += 1

        # Count unique colors
        unique_colors = len(set(individual.chromosome))

        # Fitness calculation: Negative for conflicts, subtract color count with a coefficient
        return -10 * conflicts // 2 - 3 * unique_colors

def mixed_selection(population, tournament_size):
    # Tournament selection
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda ind: ind.fitness(), reverse=True)
    winner = tournament[0]

    # Roulette wheel selection
    fitness_sum = sum(ind.fitness() for ind in population)
    if fitness_sum == 0:
        return winner
    
    pick = random.uniform(0, fitness_sum)
    current = 0
    for ind in population:
        current += ind.fitness()
        if current > pick:
            return ind

    return winner

def crossover(parent1, parent2):
    child1 = Individual(parent1.graph, parent1.num_colors)
    for i in range(len(parent1.chromosome)):
        if i < len(parent1.chromosome) / 2:
            child1.chromosome[i] = parent1.chromosome[i]
        else:
            child1.chromosome[i] = parent2.chromosome[i]
    return child1

# def mutate(individual, mutation_rate):
#     for i in range(len(individual.chromosome)):
#         if random.random() < mutation_rate:
#             individual.chromosome[i] = random.randint(0, individual.num_colors - 1)

def mutate(individual, mutation_rate):
    for i in range(len(individual.chromosome)):
        if random.random() < mutation_rate:
            current_color = individual.chromosome[i]
            new_color = random.choice([c for c in range(individual.num_colors) if c != current_color])
            individual.chromosome[i] = new_color

def genetic_algorithm(graph, num_colors, population_size, generations, elitism_size):
    # Initialize population
    population = [Individual(graph, num_colors) for _ in range(population_size)]

    for generation in range(generations):
        # Sort the population by fitness (ascending order, as fitness is negative for conflicts)
        population.sort(key=lambda ind: -ind.fitness())

        # Keep the elite individuals
        new_population = population[:elitism_size]

        # Fill the rest of the new population
        while len(new_population) < population_size:
            # Selection
            parent1 = mixed_selection(population, 5)
            parent2 = mixed_selection(population, 5)

            # Crossover
            child1 = crossover(parent1, parent2)

            # Mutation
            mutate(child1, 0.05)  # Adjust mutation rate as needed

            new_population.extend([child1])

        population = new_population[:population_size]

        # Logging
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {max([ind.fitness() for ind in population])}")

    return max(population, key=lambda ind: ind.fitness())

num_colors = 0


def read_graph_from_file(file_path):
    global num_colors
    with open(file_path, 'r') as file:
        num_nodes, num_edges = map(int, file.readline().split())
        num_colors = num_nodes
        graph = {i: [] for i in range(num_nodes + 1)}
        for line in file:
            a, b = map(int, line.split())
            graph[a].append(b)
            graph[b].append(a)  # Assuming an undirected graph
    return graph

# Example usage
file_path = 'test_5'
graph = read_graph_from_file(file_path)


# Choose one of the configurations:
population_size, generations = 200, 2000
# population_size, generations = 100, 4000
# population_size, generations = 50, 8000
elitism_size = int(population_size * 0.1)

best_individual = genetic_algorithm(graph, num_colors, population_size, generations, elitism_size)
print("Best solution:", best_individual.chromosome, best_individual.fitness())