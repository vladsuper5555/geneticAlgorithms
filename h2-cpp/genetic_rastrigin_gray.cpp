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

#define DBL_MIN -(1e10)
const int MAX_DATA_GATHERINGS = 30;
const int GEN_MAX_NO = 3000;
const double EPSILON = 0.00001;
const int POPULATION_SIZE = 200;
const double BIG_C_CONSTANT = 30000.0;

struct individ
{
    double fitness;
    std::vector<bool> *genes;
    int rank;
};

double Michalewicz(std::vector<double> inputs)
{
    double sum = 0;
    double m = 10;
    for (size_t index = 0; index < inputs.size(); index++)
    {
        double value = inputs[index];
        sum += std::sin(value) * std::pow(std::sin(((index + 1) * value * value) / M_PI), 2 * m);
    }
    return -sum;
}

// std::unordered_map<double, double> cache;
// double Schewefel(std::vector<double> inputs)
// {
//     double sum = 0;
//     for (double value : inputs)
//     {
//         auto it = cache.find(value);
//         if (it == cache.end())
//         {
//             double v = -value * std::sin(std::sqrt(std::abs(value)));
//             sum += v;
//             cache.emplace(value, v); // Use emplace instead of operator[]
//         }
//         else
//             sum += it->second; // Directly access the cached value from the iterator
//     }
//     return sum;
// }


double Schewefel (std::vector<double> inputs)
{
    double sum = 0;
    for(double value : inputs)
        sum += -value * std::sin(std::sqrt(std::abs(value)));
    return sum;
}

double Rastrigin(std::vector<double> inputs)
{
    double sum = 10 * inputs.size();
    for (double value : inputs)
    {
        sum += (value * value - 10 * std::cos(2 * M_PI * value));
    }
    return sum;
}

double De_Jong(std::vector<double> inputs)
{
    double sum = 0;
    for (double value : inputs)
    {
        sum += std::pow(value, 2);
    }
    return sum;
}

typedef double (*FunctionPointer)(std::vector<double>);

FunctionPointer functionDefinitions[] = {
    // Schewefel,
    // Michalewicz,
    Rastrigin,
    // De_Jong
};

std::pair<double, double> ranges[] = {
    // {-500.0, 500.0},
    // {0.0, M_PI},
    {-5.12, 5.12},
    // {-5.12, 5.12}
};

std::string functionNames[] = {
    // "Schewefel",
    // "Michalewicz",
    "Rastrigin",
    // "De_Jong"
};

struct Function
{
    FunctionPointer func;
    std::string name;
    std::pair<double, double> range;
};

std::vector<bool> grayToBinary(const std::vector<bool> &gray, size_t bitStringLength) {
    std::vector<bool> binary(bitStringLength, false);
    binary[0] = gray[0]; // The first bit is the same

    // Compute the rest of the bits
    for (size_t i = 1; i < bitStringLength; ++i) {
        // If current Gray code bit is 0, then copy previous binary bit, else invert it
        binary[i] = gray[i] != binary[i - 1];
    }

    return binary;
}

std::vector<double> decodeBitset(std::vector<bool> &grayBits, const std::pair<double, double> &rangeInterval, size_t bitStringLength, int dimension) {
    double rangeValue = rangeInterval.second - rangeInterval.first;

    std::vector<double> values;
    const int bitCount = dimension * bitStringLength;
    for (size_t i = 0; i < bitCount; i += bitStringLength) {
        // Convert Gray code segment to binary
        std::vector<bool> binarySegment(grayBits.begin() + i, grayBits.begin() + i + bitStringLength);
        std::vector<bool> binary = grayToBinary(binarySegment, bitStringLength);

        double value = 0;
        double range_copy = rangeValue / 2;
        for (int bit = 0; bit < bitStringLength; ++bit, range_copy /= 2) {
            value += range_copy * binary[bit];
        }
        double value_in_range = value + rangeInterval.first;
        values.push_back(value_in_range);
    }
    return values;
}

bool compareFitness(const individ &a, const individ &b)
{
    return a.fitness > b.fitness; // Sort in descending order of fitness
}

void calculateFitnessForIndivids(std::vector<individ> &fitnesses, std::vector<std::vector<bool>> &populationIndividuals, const Function &function, int bitStringLength, int dimension, double &best_function_response)
{
    for (int i = 0; i < POPULATION_SIZE; ++i)
    {
        std::vector<double> decodedBits = decodeBitset(populationIndividuals[i], function.range, bitStringLength, dimension);
        double functionValue = function.func(decodedBits);
        if (functionValue < best_function_response)
            best_function_response = functionValue;
        fitnesses[i].fitness = -functionValue + BIG_C_CONSTANT;
        fitnesses[i].genes = &populationIndividuals[i];
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

double chooseNextNeighbour(const Function &function, std::vector<bool> &currentBits, const int &method, double initialFunctionValue, const std::pair<double, double> &rangeInterval, int dimension, size_t bitStringLength)
{
    double bestValue;
    if (method == 3)
        bestValue = DBL_MIN;
    else
        bestValue = initialFunctionValue;
    bool bestNeighbour[bitStringLength * dimension];
    for (int i = 0; i < dimension * bitStringLength; ++i)
        bestNeighbour[i] = currentBits[i];

    std::vector<double> values = decodeBitset(currentBits, rangeInterval, bitStringLength, dimension);

    double initialChange = rangeInterval.second - rangeInterval.first;
    double changingValue = initialChange / 2;
    int value_position = -1;
    const int bitCount = bitStringLength * dimension;
    for (size_t index = 0; index < bitCount; ++index)
    {
        if (index % bitStringLength == 0)
        {
            changingValue = initialChange / 2;
            value_position++;
        }
        if (currentBits[index] == 1)
            values[value_position] -= changingValue;
        else
            values[value_position] += changingValue;
        currentBits[index] = !currentBits[index];
        double functionValue = function.func(values);

        if (method == 1 && functionValue < bestValue)
        {
            bestValue = functionValue;

            for (int i = 0; i < dimension * bitStringLength; ++i)
                bestNeighbour[i] = currentBits[i];
        }
        if (currentBits[index] == 1)
            values[value_position] -= changingValue;
        else
            values[value_position] += changingValue;
        currentBits[index] = !currentBits[index];
        changingValue /= 2;
    }
    if (bestValue == DBL_MIN)
        return initialFunctionValue;

    for (int i = 0; i < dimension * bitStringLength; ++i)
        currentBits[i] = bestNeighbour[i];
    return bestValue;
}

double hill_climb_algorithm(const Function &function, int dimension, const int bitStringLength, std::vector<bool> currentBits)
{
    double current_value = function.func(decodeBitset(currentBits, function.range, bitStringLength, dimension));
    bool local = false;
    double best_function_response = std::numeric_limits<double>::infinity();
    while (!local)
    {
        auto nextNeighbourValue = chooseNextNeighbour(function, currentBits, 1, current_value, function.range, dimension, bitStringLength);
        // std::cout << "Going to try and improve " << current_value << " with " << nextNeighbourValue << '\n';
        if (nextNeighbourValue < current_value)
            current_value = nextNeighbourValue;
        else
            local = true;
    }

    if (current_value < best_function_response)
    {
        best_function_response = current_value;
    }

    return best_function_response;
}

std::pair<std::vector<double>, std::chrono::duration<double>> genetic(const Function &function, int dimension)
{
    size_t bitStringLength = std::ceil(std::log2((function.range.second - function.range.first) / EPSILON));

    std::vector<double> results;
    std::mutex results_mutex; // Mutex to control access to the results vector
    int num_threads = std::thread::hardware_concurrency();

    auto worker = [&](int start, int end)
    {
        // std::cout << "workier from " << start << " to " << end << '\n';
        std::mt19937_64 gen(static_cast<std::mt19937_64::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        gen.seed(gen());
        for (int test = start; test < end; ++test)
        {
            gen.seed(gen());
            double best_function_response = std::numeric_limits<double>::infinity();

            std::vector<std::vector<bool>> population(POPULATION_SIZE);
            std::uniform_int_distribution<> distrib(0, 1);
            for (int i = 0; i < POPULATION_SIZE; ++i)
                for (size_t j = 0; j < bitStringLength * dimension; ++j)
                    population[i].emplace_back(distrib(gen) % 2);

            for (int i = 0; i < GEN_MAX_NO; ++i)
            {
                std::vector<individ> fitnesses(POPULATION_SIZE);
                calculateFitnessForIndivids(fitnesses, population, function, bitStringLength, dimension, best_function_response);
                // since now we have the results from the previous generation lets calculate the best
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
                    for (int jj = 0; jj < dimension * bitStringLength; ++jj)
                        v.push_back((*(fitnesses[ii].genes))[jj]);
                    newPopulation.push_back(v);
                }

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

                    std::vector<bool> child(bitStringLength * dimension);
                    for (int jj = 0; jj < bitStringLength * dimension; ++jj)
                        child[jj] = population[indv1][jj];

                    newPopulation.push_back(child);
                    generatedPopCount++;
                }
                // CROSSOVER
                while (generatedPopCount < POPULATION_SIZE)
                {
                    // Select two individuals using the roulette wheel selection
                    int indv1 = rouletteWheelSelection(fitnesses);
                    int indv2 = rouletteWheelSelection(fitnesses);

                    // Make sure we have two different individuals to cross
                    if (indv1 == indv2)
                        continue;

                    // Crossover to produce a new individual
                    std::vector<bool> child(bitStringLength * dimension);
                    crossover(population[indv1], population[indv2], child, bitStringLength * dimension);

                    // Add the child to the new population
                    newPopulation.push_back(child);
                    generatedPopCount++;
                }

                // we just need mutation and done
                for (int ii = elitismCount; ii < POPULATION_SIZE; ++ii)
                    // we go through each bit and try and mutate it
                    for (int jj = 0; jj < bitStringLength * dimension; ++jj)
                    {
                        std::uniform_real_distribution<double> dis(0.0, 1.0 - EPSILON);
                        double randomNumber = dis(gen);
                        if (randomNumber < 1.2 / (dimension * bitStringLength))
                            newPopulation[ii][jj] = !newPopulation[ii][jj];
                    }
                population = std::move(newPopulation);
            }
            for (int ii = 0; ii < POPULATION_SIZE; ++ii)
                best_function_response = std::min(best_function_response, hill_climb_algorithm(function, dimension, bitStringLength, population[ii]));
            {
                std::lock_guard<std::mutex> lock(results_mutex); // Lock the mutex while modifying shared resources
                results.push_back(best_function_response);
            }
        }
    };

    std::vector<std::thread> threads;
    int tests_per_thread = std::max(1, MAX_DATA_GATHERINGS / num_threads);
    auto start_time = std::chrono::high_resolution_clock::now();


    for (int i = 0; i < num_threads; ++i)
    {
        int start = i * tests_per_thread;
        int end = (i + 1) * tests_per_thread;

        // For the last thread, ensure it processes all remaining tests
        if (i == num_threads - 1)
            end = MAX_DATA_GATHERINGS;

        threads.emplace_back(worker, start, end);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time);

    return {results, time_span / num_threads}; // else this is not the paralele time execution but the total CPU time
}

int main(int argc, char *argv[])
{
    std::srand(std::time(0));

    std::vector<Function> functions;
    for (size_t i = 0; i < sizeof(functionDefinitions) / sizeof(FunctionPointer); ++i)
    {
        functions.push_back({functionDefinitions[i], functionNames[i], ranges[i]});
    }

    std::vector<int> dimensions = {5, 10, 30};

    std::string filename = "genetic.txt";
    std::ofstream output_file(filename);

    for (const auto &function : functions)
    {
        for (const auto &dimension : dimensions)
        {
            auto res = genetic(function, dimension);
            double sum = 0;
            double sum_time = 0;
            double best = 100000;
            //  std::cout << r << '\n',
            double variance = 0;
            for (auto r : res.first)
                sum += r, best = std::min(best, r);
            const double median = sum / res.first.size();
            for (auto r : res.first)
                variance += std::pow(median - r, 2);
            variance = variance / res.first.size();
            const double std_dev = std::sqrt(variance);
            // std::cout << sum / res.first.size() << ' ' << best << ' ' << res.second.count() << '\n';
            std::cout << "Function: " << function.name << ", Dimension: " << dimension << ", Min Value: " << std::fixed << std::setprecision(5) << best << ", Median " << median << ", Std. dev. " << std_dev << ", Time: " << res.second.count() << std::endl;
        }
    }

    output_file.close();
    return 0;
}
// while crossover rate goes down the mutation probability should go up because at some point in futre time we will have very close individuals
// this means that crcossover is not really helpful but the other is 
// mutation reate should be inscresing or decresing
// trigger hypermutation cobb
// diversity of solutions decresing is bad
// should check the last time we improved the algothim

// if the generation stats being the same we can generate some random new population members (aroudn the change of changin a bit is 40%)