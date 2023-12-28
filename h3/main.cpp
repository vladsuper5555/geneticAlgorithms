#include <iostream>
#include <cmath>
#include <map>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <set>
using namespace std;
#define POPULATION_SIZE 50
#define GENERATION_COUNT 8000
// #define ELITES_COUNT 3
// #define TOURNAMENT_SELECTION 17
// #define CROSSOVER_COUNT 30
#define NUMBER 2.715634
int ELITES_COUNT, TOURNAMENT_SELECTION, CROSSOVER_COUNT;
int next_individuals_from_selction_number;
vector<int> g[1000];
int nodes, edges;
int bits_for_node_encoding;
int total_bits_for_gene;
struct individ
{
    bool chromosome[500 * 10]; // thsi should be enough for endoding of a gene
    int current_fitness;
};
void generateRandomPopulation(vector<individ> &population)
{
    mt19937_64 gen(static_cast<std::mt19937_64::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    gen.seed(gen());
    std::uniform_int_distribution<int> distrib(0, 1);
    for (int i = 0; i < POPULATION_SIZE; ++i)
        for (int j = 0; j < total_bits_for_gene; ++j)
            population[i].chromosome[j] = distrib(gen);
}
int evaluateIndividual(vector<int> decodedValues)
{
    int plusPerMissingColor = 1;
    map<int, int> fr;
 
    int errors_count = 0;
    int number_of_unique_colors = 0;
 
    for (int i = 0; i < nodes; ++i)
        fr[decodedValues[i]] = 1;
    for (auto it : fr)
        number_of_unique_colors++;
    int fitness = (nodes - number_of_unique_colors) * plusPerMissingColor;
 
    return fitness;
}
void decodeValues(individ &gene, vector<int> &storeValues)
{
    for (int i = 0; i < total_bits_for_gene; i += bits_for_node_encoding)
    {
        int valueToPush = 0;
        for (int j = i; j < i + bits_for_node_encoding; ++j)
            valueToPush = (valueToPush << 1) | gene.chromosome[j];
        storeValues.emplace_back(valueToPush % nodes); // in order to reduce automatically the numbere of colors // because of the ceil
    }
}
void evaluateGeneration(vector<individ> &population)
{
    for (int i = 0; i < POPULATION_SIZE; ++i)
    {
        vector<int> decodedValues;
        decodeValues(population[i], decodedValues);
        population[i].current_fitness = evaluateIndividual(decodedValues);
    }
}
bool cmp(individ &a, individ &b)
{
    return a.current_fitness > b.current_fitness;
}
void selectRemainingPopulation(vector<individ> &newPopulation, vector<individ> &population)
{
    // first lets do a simple selection just by taking the best individuals
    // will use a higher mutation rate for the eventual problems
    // first lets sort the population
    // lets make a tournament selection for the other half
    mt19937_64 gen(static_cast<std::mt19937_64::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    gen.seed(gen());
    sort(population.begin(), population.end(), cmp);
    for (int i = 0; i < ELITES_COUNT; ++i)
        newPopulation[i] = population[i];
    // tournament selection
    for (int i = 0; i < TOURNAMENT_SELECTION; ++i)
    {
        int minIndex = 10000;
        for (int j = 1; j <= 3 /*the number of selected individuals*/; ++j) // normal results with 15
        {
            std::uniform_int_distribution<int> distrib(ELITES_COUNT, POPULATION_SIZE - 1);
            int generatedIndex = distrib(gen);
            minIndex = min(minIndex, generatedIndex);
        }
        newPopulation[ELITES_COUNT + i] = population[minIndex];
    }
}
void crossoverNewPopulation(vector<individ> &population)
{
    // we need to add the diffrnce by crossover
    mt19937_64 gen(static_cast<std::mt19937_64::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    gen.seed(gen());
    for (int i = 0; i < CROSSOVER_COUNT; ++i)
    {
        std::uniform_int_distribution<int> distrib(0, next_individuals_from_selction_number + i - 1);
        int indexToSaveTo = next_individuals_from_selction_number + i;
        // now we need to choose two indivudals
        int individual1 = distrib(gen);
        int individual2 = distrib(gen);
        while (individual1 == individual2)
            individual2 = distrib(gen);
 
        int individual3 = distrib(gen);
        while (individual3 == individual2 || individual3 == individual1)
            individual3 = distrib(gen);
 
        int individual4 = distrib(gen);
        while (individual4 == individual3 || individual3 == individual2 || individual4 == individual1)
            individual4 = distrib(gen);
 
        if (population[individual2].current_fitness > population[individual1].current_fitness)
            swap(individual1, individual2);
 
        if (population[individual4].current_fitness > population[individual3].current_fitness)
            swap(individual3, individual4);
 
        swap(individual3, individual2);
        uniform_int_distribution<> d(0, total_bits_for_gene - 1);
        int point1 = d(gen);
 
        for (int i = 0; i < total_bits_for_gene; ++i)
            if (i < point1)
                population[indexToSaveTo].chromosome[i] = population[individual1].chromosome[i];
            else
                population[indexToSaveTo].chromosome[i] = population[individual2].chromosome[i];
    }
}
void mutateNewPopulationByRandomBits(vector<individ> &population)
{
    mt19937_64 gen(static_cast<std::mt19937_64::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    gen.seed(gen());
    float prob = 2.5 / total_bits_for_gene;
    uniform_real_distribution<double> distrib(0, 1.0 - 0.000001);
    for (int i = ELITES_COUNT; i < POPULATION_SIZE; ++i)
        for (int j = 0; j < total_bits_for_gene; ++j)
            if (distrib(gen) < prob) // 1.2 % for each gene
                population[i].chromosome[j] = !population[i].chromosome[j];
}
 
void encodeValuesInIndividual(individ &i, vector<int> &colors)
{
    int index = 0;
    for (auto color : colors)
    {
        int next_minus = (1 << (bits_for_node_encoding - 1));
        for (int j = 0; j < bits_for_node_encoding; ++j)
        {
            // printf("the color is %d and the minus is %d\n", color, next_minus);
            if (color >= next_minus)
            {
                color -= next_minus;
                i.chromosome[index] = 1;
            }
            else
                i.chromosome[index] = 0;
            index++;
            next_minus >>= 1;
        }
    }
}
 
void mutateNewPopulation(vector<individ> &population)
{
    mt19937_64 gen(static_cast<std::mt19937_64::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    gen.seed(gen());
    uniform_int_distribution<int> distrib(0, nodes - 1);
    for (auto &indv : population)
    {
        vector<int> colors;
        decodeValues(indv, colors); // maybe a random check before chaning the color
        for (int node = 0; node < nodes; ++node)
        {
            for (auto neigh : g[node])
                if (colors[node] == colors[neigh])
                {
                    // here we need toi change its color;
                    set<int> unavailableColors;
                    for (auto n : g[node])
                        unavailableColors.insert(n);
                    while (1)
                    {
                        int i = distrib(gen);
                        if (unavailableColors.find(i) == unavailableColors.end())
                        {
                            colors[node] = i;
                            break;
                        }
                    }
                    continue;
                }
        }
        encodeValuesInIndividual(indv, colors);
        // encode the individual back
    }
}
 
int outputResults(vector<individ> &population)
{
    int MIN_COLORS = 100000;
    for (int i = 0; i < POPULATION_SIZE; ++i)
        MIN_COLORS = min(MIN_COLORS, nodes - population[i].current_fitness);
    return MIN_COLORS;
}
 
void improveWithSimulationgAnnealingLastPopulation(vector<individ> &population)
{
    std::mt19937_64 gen(static_cast<std::mt19937_64::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    gen.seed(gen());
    std::uniform_int_distribution<> distrib(0, 1);
    auto start_time = std::chrono::high_resolution_clock::now();
 
    double MAX_ITERATIONS = 25;
    int index = 0;
    for (auto &indv : population)
    {   index++;
        double best_function_response = indv.current_fitness;
 
        double T = 100; // this is the temperature
 
        for (int i = 0; i < 2500; ++i) // this is for the temparature
        {
            int not_improved_iterations = 0;
            while (not_improved_iterations < (int)MAX_ITERATIONS)
            {
                std::uniform_real_distribution<double> dis(0.0, 1.0 - 0.00001);
                double random_number = dis(gen);
                if (random_number < T / 100.0)
                {
                    vector<int> intialColors;
                    decodeValues(indv, intialColors);
 
                    for (int k = 0; k < total_bits_for_gene / bits_for_node_encoding; ++k)
                    {
                        // probably better to change in eveery node node at random
                        std::uniform_int_distribution<> bitDistrib(k * bits_for_node_encoding, (k + 1) * bits_for_node_encoding - 1);
                        int randomBit = bitDistrib(gen);
                        indv.chromosome[randomBit] = !indv.chromosome[randomBit];
                        randomBit = bitDistrib(gen);
                        indv.chromosome[randomBit] = !indv.chromosome[randomBit];
                    }
                    // for (int i = 0; i < nodes / 50; ++i)
                    // {
                    //     std::uniform_int_distribution<> bitDistrib(0, total_bits_for_gene);
                    //     int randomBit = bitDistrib(gen);
                    //     indv.chromosome[randomBit] = !indv.chromosome[randomBit];
                    // }
                    // cout << "changed at random\n" << random_number << " " << std::exp(1.0 * -(abs(nextNeighbourValue - best_function_response + (nextNeighbourValue == best_function_response ? 10 : 0)) / T)) << '\n';
                    // basically making sure we still have good values
 
                    vector<int> colors;
                    decodeValues(indv, colors);
                    std::uniform_int_distribution<> d(0, nodes - 1);
                    for (int node = 0; node < nodes; ++node)
                    {
                        for (auto neigh : g[node])
                            if (colors[node] == colors[neigh])
                            {
                                // here we need toi change its color;
                                set<int> unavailableColors;
                                for (auto n : g[node])
                                    unavailableColors.insert(n);
                                while (1)
                                {
                                    int i = d(gen);
                                    if (unavailableColors.find(i) == unavailableColors.end())
                                    {
                                        colors[node] = i;
                                        break;
                                    }
                                }
                                continue;
                            }
                    }
                    encodeValuesInIndividual(indv, colors);
                    indv.current_fitness = evaluateIndividual(colors);
                    if (indv.current_fitness > best_function_response)
                    {
                        best_function_response = indv.current_fitness;
                        not_improved_iterations = 0;
                    }
                    else
                    {
                        encodeValuesInIndividual(indv, intialColors);
                        indv.current_fitness = evaluateIndividual(intialColors);
                        not_improved_iterations++;
                    }
                }
                not_improved_iterations++;
            }
            T = T * 0.995;
            // printf("The temperature is %f\n", T);
        }
    }
}
 
int main(int argc, char **argv)
{
    ELITES_COUNT = atoi(argv[1]);
    TOURNAMENT_SELECTION = atoi(argv[2]);
    CROSSOVER_COUNT = POPULATION_SIZE - ELITES_COUNT - TOURNAMENT_SELECTION;
    next_individuals_from_selction_number = ELITES_COUNT + TOURNAMENT_SELECTION;
    ifstream in("test_10");
    in >> nodes >> edges;
    for (int i = 0; i < edges; ++i)
    {
        int a, b;
        in >> a >> b;
        g[a].push_back(b);
        g[b].push_back(a);
    }
    // first lets generate a population
    int generation_count = 30;
    int total = 0;
    for (int i = 0; i < generation_count; ++i)
    {
        vector<individ> population(POPULATION_SIZE);
        bits_for_node_encoding = ceil(log2(1.0 * nodes));
        total_bits_for_gene = nodes * bits_for_node_encoding;
        generateRandomPopulation(population);
        for (int generation = 1; generation <= GENERATION_COUNT; ++generation)
        {
            evaluateGeneration(population);
            vector<individ> newPopulation(POPULATION_SIZE);
            selectRemainingPopulation(newPopulation, population);
            crossoverNewPopulation(newPopulation);
            mutateNewPopulationByRandomBits(newPopulation); // this mutates for exploratiopn
            mutateNewPopulation(newPopulation);             // this mutates for explatation
            // if (generation % 1000 == 0)
            //     improveWithSimulationgAnnealingLastPopulation(newPopulation);
            population = move(newPopulation);
        }
        // evaluateGeneration(population);
        // improveWithSimulationgAnnealingLastPopulation(population);
        evaluateGeneration(population);
        total += outputResults(population);
    }
    cout << "The average is " << total / generation_count << '\n';
    ofstream out ("average.out");
    out << 1.0 * total / generation_count;
    return total;
}
// implement simulating annealing for choosiong the parameters values 
// like in the metaheuristic for ga but put it simply in a sa and have a best function response the minimum after running 8000 geneartions
// then just read the parameters and use them as (found values)