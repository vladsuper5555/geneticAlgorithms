#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <bitset>
#include <string>
#include <chrono>
#include <random>
#include <unordered_map>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <limits.h>
#include <thread>
#include <mutex>
#include <map>

#define DBL_MIN -(1e10)
const int MAX_DATA_GATHERINGS = 3;
const int GEN_MAX_NO = 10000;
const double EPSILON = 1;
const int POPULATION_SIZE = 1000;
const double BIG_C_CONSTANT = 30000.0;
int nodes, edges;
std::vector<int> graph[10001];
struct individ
{
    double fitness;
    std::vector<bool> *genes;
    int rank;
};

std::vector<int> decodeBitset(std::vector<bool> &bits, size_t bitStringLength)
{
    std::vector<int> values;
    const int bitCount = bitStringLength * nodes;
    for (int i = 0; i < bitCount; i += bitStringLength)
    {
        int value = 0;
        for (int bit = i; bit < i + bitStringLength; ++bit)
            value = value * 2 + bits[bit];
        values.push_back(value);
    }
    return values;
}

bool compareFitness(const individ &a, const individ &b)
{
    return a.fitness > b.fitness; // Sort in descending order of fitness
}
// here we shoudl still check if the graph is connected
int evaluateColoringOnGraph (std::vector<int>& coloring) 
{
    // std::cout << "the coloring\n";
    // for (auto it : coloring)
    //     std::cout << it << ' ';
    // std::cout << '\n';
    // maybe a better way to evaluate is to make the difference between the number of
    // well colored nodes and the ones that are not well colored
    // std::cout << "new graph\n";
    for (int node = 0; node < nodes; ++node)
    {
        for (auto vecin : graph[node])
        {
            // std::cout << "my color is " << coloring[node] << "my neighbour is " << coloring[vecin] << '\n';
            if (coloring[node] == coloring[vecin])
                return 1;
            }
    }
    int cnt = 1;
    std::map <int, int> fr;
    // calculating the number of unique colors in the graph
    // for (auto it : coloring)
    //     std::cout << it << ' ';
    // std::cout << '\n';
    for (auto it : coloring)
        fr[it]++;
    for (auto it : fr)
        cnt++;
    // std::cout << "the count is " << cnt << '\n';
    return cnt;
}

void calculateFitnessForIndivids(std::vector<individ> &fitnesses, std::vector<std::vector<bool>> &populationIndividuals, int bitStringLength, int &best_function_response)
{
    for (int i = 0; i < POPULATION_SIZE; ++i)
    {
        std::vector<int> decodedBits = decodeBitset(populationIndividuals[i], bitStringLength);
        int functionValue = evaluateColoringOnGraph(decodedBits);
        // std::cout << "the evaluated function value is : " << functionValue << '\n';
        if (functionValue < best_function_response && functionValue != 1)
            best_function_response = functionValue;
        fitnesses[i].fitness = functionValue;
        // fitnesses[i].fitness = functionValue + BIG_C_CONSTANT;
        fitnesses[i].genes = &populationIndividuals[i];
        // std::cout << "Evaluated the individual " << i << '\n';
    }
}

void crossover(const std::vector<bool> &indv1, const std::vector<bool> &indv2, std::vector<bool> &child, int genes)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, genes - 2); // Ensure cutting points are within bounds

    int cut1 = dist(gen);
    int cut2 = dist(gen);
    int cut3 = dist(gen);

    // Ensure the cutting points are in ascending order
    if (cut1 > cut2)
        std::swap(cut1, cut2);
    if (cut2 > cut3)
        std::swap(cut2, cut3);
    if (cut1 > cut2)
        std::swap(cut1, cut2);

    // Perform crossover using the three cutting points
    for (int i = 0; i < genes; ++i)
    {
        if (i < cut1 || (i >= cut2 && i < cut3))
            child[i] = indv1[i];
        else
            child[i] = indv2[i];
    }
}

int tournamentSelection(const std::vector<individ> &fitnesses, int tournamentSize)
{
    std::mt19937 gen(std::random_device{}());
    gen.seed(gen());
    std::uniform_int_distribution<> dist(0, fitnesses.size() - 1);

    int best = dist(gen);
    for (int i = 1; i < tournamentSize; ++i)
    {
        int next = dist(gen);
        if (fitnesses[next].fitness > fitnesses[best].fitness)
            best = next;
    }
    return best;
}

int rouletteWheelSelection(const std::vector<individ> &fitnesses)
{
    std::mt19937_64 gen(static_cast<std::mt19937_64::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    gen.seed(gen());
    double totalFitness = 0;
    for (const auto &ind : fitnesses)
        totalFitness += ind.fitness;

    std::uniform_real_distribution<double> distrib(0, totalFitness);
    double slice = distrib(gen);

    double fitnessSum = 0;
    for (size_t i = 0; i < fitnesses.size(); ++i)
    {
        fitnessSum += fitnesses[i].fitness;
        if (fitnessSum >= slice)
            return i;
    }

    return fitnesses.size() - 1;
}

// double chooseNextNeighbour(std::vector<bool> &currentBits, const int &method, double initialFunctionValue, const std::pair<double, double> &rangeInterval, int dimension, size_t bitStringLength)
// {
//     double bestValue;
//     if (method == 3)
//         bestValue = DBL_MIN;
//     else
//         bestValue = initialFunctionValue;
//     bool bestNeighbour[bitStringLength * dimension];
//     for (int i = 0; i < dimension * bitStringLength; ++i)
//         bestNeighbour[i] = currentBits[i];

//     std::vector<double> values = decodeBitset(currentBits, rangeInterval, bitStringLength, dimension);

//     double initialChange = rangeInterval.second - rangeInterval.first;
//     double changingValue = initialChange / 2;
//     int value_position = -1;
//     const int bitCount = bitStringLength * dimension;
//     for (size_t index = 0; index < bitCount; ++index)
//     {
//         if (index % bitStringLength == 0)
//         {
//             changingValue = initialChange / 2;
//             value_position++;
//         }
//         if (currentBits[index] == 1)
//             values[value_position] -= changingValue;
//         else
//             values[value_position] += changingValue;
//         currentBits[index] = !currentBits[index];
//         double functionValue = function.func(values);

//         if (method == 1 && functionValue < bestValue)
//         {
//             bestValue = functionValue;

//             for (int i = 0; i < dimension * bitStringLength; ++i)
//                 bestNeighbour[i] = currentBits[i];
//         }
//         if (currentBits[index] == 1)
//             values[value_position] -= changingValue;
//         else
//             values[value_position] += changingValue;
//         currentBits[index] = !currentBits[index];
//         changingValue /= 2;
//     }
//     if (bestValue == DBL_MIN)
//         return initialFunctionValue;

//     for (int i = 0; i < dimension * bitStringLength; ++i)
//         currentBits[i] = bestNeighbour[i];
//     return bestValue;
// }

// double hill_climb_algorithm(int dimension, const int bitStringLength, std::vector<bool> currentBits)
// {
//     double current_value = function.func(decodeBitset(currentBits, function.range, bitStringLength, dimension));
//     bool local = false;
//     double best_function_response = std::numeric_limits<double>::infinity();
//     while (!local)
//     {
//         auto nextNeighbourValue = chooseNextNeighbour(function, currentBits, 1, current_value, function.range, dimension, bitStringLength);
//         // std::cout << "Going to try and improve " << current_value << " with " << nextNeighbourValue << '\n';
//         if (nextNeighbourValue < current_value)
//             current_value = nextNeighbourValue;
//         else
//             local = true;
//     }

//     if (current_value < best_function_response)
//     {
//         best_function_response = current_value;
//     }

//     return best_function_response;
// }

std::pair<std::vector<int>, std::chrono::duration<double>> genetic()
{
    size_t bitStringLength = std::ceil(std::log2((1.0 * nodes / EPSILON))); // this is the number of bits for one node

    std::vector<int> results;
    std::mutex results_mutex; // Mutex to control access to the results vector
    // int num_threads = std::thread::hardware_concurrency();
    auto start_time = std::chrono::high_resolution_clock::now();

    // auto worker = [&](int start, int end)
    // {
        // std::cout << "workier from " << start << " to " << end << '\n';
        std::mt19937_64 gen(static_cast<std::mt19937_64::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        gen.seed(gen());
        for (int test = 1; test < MAX_DATA_GATHERINGS; ++test)
        {
            gen.seed(gen());
            int best_function_response = 500000;
            std::vector<std::vector<bool>> population(POPULATION_SIZE);
            std::uniform_int_distribution<> distrib(0, 1);
            for (int i = 0; i < POPULATION_SIZE; ++i)
                for (size_t j = 0; j < bitStringLength * nodes; ++j)
                    population[i].emplace_back(distrib(gen) % 2);

            for (int i = 0; i < GEN_MAX_NO; ++i)
            {
                std::vector<individ> fitnesses(POPULATION_SIZE);
                calculateFitnessForIndivids(fitnesses, population, bitStringLength, best_function_response);
                // since now we have the results from the previous generation lets calculate the best
                // std::cout << "evaluated the population \n";
                std::sort(fitnesses.begin(), fitnesses.end(), compareFitness);
                for (int ii = 0; ii < fitnesses.size(); ++ii)
                {
                    fitnesses[ii].rank = ii + 1;
                    // if (ii == 0)
                    //     std::cout << "The fitness for the gen: " << i << " is " << -fitnesses[ii].fitness + BIG_C_CONSTANT  << '\n';
                }
                std::vector<std::vector<bool>> newPopulation;

                int generatedPopCount = POPULATION_SIZE * 0.15;
                int elitismCount = POPULATION_SIZE * 0.15;
                for (int ii = 0; ii < elitismCount; ++ii)
                {
                    std::vector<bool> v;
                    for (int jj = 0; jj < nodes * bitStringLength; ++jj)
                        v.push_back((*(fitnesses[ii].genes))[jj]);
                    newPopulation.push_back(v);
                }
                // std::cout << "evalutated the elites\n";
                // while (generatedPopCount < POPULATION_SIZE * 0.3)
                // {
                //     int indv1 = rouletteWheelSelection(fitnesses);

                //     std::vector<bool> child(bitStringLength * dimension);
                //     for (int jj = 0; jj < bitStringLength * dimension; ++jj)
                //         child[jj] = population[indv1][jj];

                //     newPopulation.push_back(child);
                //     generatedPopCount++;
                // }

                while (generatedPopCount < POPULATION_SIZE * 0.4) // 0.5 very good 0.4 seems even better
                {
                    int indv1 = tournamentSelection(fitnesses, 10); // Example tournament size of 5

                    std::vector<bool> child(bitStringLength * nodes);
                    for (int jj = 0; jj < bitStringLength * nodes; ++jj)
                        child[jj] = population[indv1][jj];

                    newPopulation.push_back(child);
                    generatedPopCount++;
                }
                // std::cout << "evaluted the tournament\n";
                // CROSSOVER
                // for (auto it : fitnesses)
                //     std::cout << it.fitness << '\n';
                while (generatedPopCount < POPULATION_SIZE)
                {
                    // Select two individuals using the roulette wheel selection
                    int indv1 = rouletteWheelSelection(fitnesses);
                    int indv2 = rouletteWheelSelection(fitnesses);

                    // Make sure we have two different individuals to cross
                    if (indv1 == indv2)
                        continue;

                    // Crossover to produce a new individual
                    std::vector<bool> child(bitStringLength * nodes);
                    crossover(population[indv1], population[indv2], child, bitStringLength * nodes);

                    // Add the child to the new population
                    newPopulation.push_back(child);
                    generatedPopCount++;
                }
                // std::cout << "we have a crossover for the new populatii\n";
                // we just need mutation and done
                for (int ii = elitismCount; ii < POPULATION_SIZE; ++ii)
                    // we go through each bit and try and mutate it
                    for (int jj = 0; jj < bitStringLength * nodes; ++jj)
                    {
                        std::uniform_real_distribution<double> dis(0.0, 1.0 - 0.0001);
                        double randomNumber = dis(gen);
                        if (randomNumber < 20 / (i * nodes * bitStringLength))
                            newPopulation[ii][jj] = !newPopulation[ii][jj];
                    }
                // std::cout << "the new population is mutated\n";
                population = std::move(newPopulation);
                // std::cout << best_function_response << '\n';
            }
            // for (int ii = 0; ii < POPULATION_SIZE; ++ii)
            //     best_function_response = std::min(best_function_response, hill_climb_algorithm(function, dimension, bitStringLength, population[ii]));
            // {
                // std::lock_guard<std::mutex> lock(results_mutex); // Lock the mutex while modifying shared resources
                results.push_back(best_function_response - 1);
            // }
        }
    // };

    // std::vector<std::thread> threads;
    // int tests_per_thread = std::max(1, MAX_DATA_GATHERINGS / num_threads);


    // for (int i = 0; i < num_threads; ++i)
    // {
    //     int start = i * tests_per_thread;
    //     int end = (i + 1) * tests_per_thread;

    //     // For the last thread, ensure it processes all remaining tests
    //     if (i == num_threads - 1)
    //         end = MAX_DATA_GATHERINGS;

    //     threads.emplace_back(worker, start, end);
    // }

    // for (auto &thread : threads)
    // {
    //     thread.join();
    // }

    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time);

    return {results, time_span}; // else this is not the paralele time execution but the total CPU time
}

int main(int argc, char *argv[])
{
    std::srand(std::time(0));

    // first lets read the graph
    std::ifstream in("test1.col");
    in >> nodes >> edges;
    for (int i = 1; i <= edges; ++i) {
        int a, b;
        in >> a >> b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }

    auto res = genetic();
    std::cout << "the results are\n";
    for (auto it : res.first)
        std::cout << it << '\n';
    return 0;
}
// while crossover rate goes down the mutation probability should go up because at some point in futre time we will have very close individuals
// this means that crcossover is not really helpful but the other is 
// mutation reate should be inscresing or decresing
// trigger hypermutation cobb
// diversity of solutions decresing is bad
// should check the last time we improved the algothim

// if the generation stats being the same we can generate some random new population members (aroudn the change of changin a bit is 40%)