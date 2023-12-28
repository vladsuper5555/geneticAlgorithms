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
#include <set>

#define DBL_MIN -(1e10)
const int MAX_DATA_GATHERINGS = 3;
const int GEN_MAX_NO = 2000;
const double EPSILON = 1;
const int POPULATION_SIZE = 200;
const double BIG_C_CONSTANT = 30000.0;
int nodes, edges;
std::vector<int> graph[10001];
struct individ
{
    double fitness;
    std::vector<bool> *genes;
    int rank;
};


int evaluateColoringOnGraph(const std::vector<int>& coloring, int & best_function_response) {
    int colorViolations = 0;
    std::set<int> uniqueColors;

    // Check for color violations and count unique colors
    for (int node = 0; node < nodes; ++node) {
        uniqueColors.insert(coloring[node]);
        for (int neighbor : graph[node]) {
            if (coloring[node] == coloring[neighbor]) {
                colorViolations++;
            }
        }
    }

    // Calculate the fitness score
    int numberOfUniqueColors = uniqueColors.size();
    double fitnessScore;

    // Base fitness value - should be larger than the maximum possible penalty
    const double BASE_FITNESS = nodes * 105 + edges * 55; // This value can be adjusted as needed
    const double PENALTY_PER_VIOLATION = 50; // Adjust based on the expected range of violations
    const double PENALTY_PER_COLOR = 100;
    // Calculate fitness score
    fitnessScore = BASE_FITNESS - (colorViolations * PENALTY_PER_VIOLATION) - numberOfUniqueColors * PENALTY_PER_COLOR;
    if (colorViolations == 0 && numberOfUniqueColors < best_function_response)
        best_function_response = numberOfUniqueColors;
    return fitnessScore;
}

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

double chooseNextNeighbour(std::vector<bool> &currentBits, double initialFunctionValue, size_t bitStringLength, int & best_function_response)
{
    double bestValue = initialFunctionValue;
    bool bestNeighbour[bitStringLength * nodes];
    for (int i = 0; i < nodes * bitStringLength; ++i)
        bestNeighbour[i] = currentBits[i];


    for (size_t index = 0; index < bitStringLength * nodes; ++index)
    {
        currentBits[index] = !currentBits[index];
        std::vector<int> values = decodeBitset(currentBits, bitStringLength);
        double functionValue = evaluateColoringOnGraph(values, best_function_response);

        if (functionValue < bestValue)
        {
            bestValue = functionValue;

            for (int i = 0; i < nodes * bitStringLength; ++i)
                bestNeighbour[i] = currentBits[i];
        }
        currentBits[index] = !currentBits[index];
    }

    for (int i = 0; i < nodes * bitStringLength; ++i)
        currentBits[i] = bestNeighbour[i];
    return bestValue;
}

void hill_climb_algorithm(const int bitStringLength, std::vector<bool> currentBits, int & best_function_response)
{
    double current_value = evaluateColoringOnGraph(decodeBitset(currentBits, bitStringLength), best_function_response);
    bool local = false;
    while (!local)
    {
        auto nextNeighbourValue = chooseNextNeighbour(currentBits, current_value, bitStringLength, best_function_response);
        // std::cout << "Going to try and improve " << current_value << " with " << nextNeighbourValue << '\n';
        if (nextNeighbourValue < current_value)
            current_value = nextNeighbourValue;
        else
            local = true;
    }

}


bool compareFitness(const individ &a, const individ &b)
{
    return a.fitness > b.fitness; // Sort in descending order of fitness
}


void calculateFitnessForIndivids(std::vector<individ> &fitnesses, std::vector<std::vector<bool>> &populationIndividuals, int bitStringLength, int &best_function_response)
{
    for (int i = 0; i < POPULATION_SIZE; ++i)
    {
        std::vector<int> decodedBits = decodeBitset(populationIndividuals[i], bitStringLength);
        int functionValue = evaluateColoringOnGraph(decodedBits, best_function_response);
        // if (functionValue < best_function_response && functionValue != 1)
        //     best_function_response = functionValue;
        fitnesses[i].fitness = functionValue;
        fitnesses[i].genes = &populationIndividuals[i];
    }
}

// void crossover(const std::vector<bool> &indv1, const std::vector<bool> &indv2, std::vector<bool> &child, int genes)
// {
//     std::mt19937 gen(std::random_device{}());
//     std::uniform_int_distribution<int> dist(1, genes - 2); // Ensure the cutting point is within bounds

//     int cutPoint = dist(gen); // Single cutting point

//     // Perform crossover using the single cutting point
//     for (int i = 0; i < genes; ++i)
//     {
//         if (i < cutPoint)
//             child[i] = indv1[i]; // Copy from first parent
//         else
//             child[i] = indv2[i]; // Copy from second parent
//     }
// }

void crossover(const std::vector<bool> &indv1, const std::vector<bool> &indv2, std::vector<bool> &child, int genes)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, genes - 2); // Ensure cutting points are within bounds

    // Generate four unique cutting points
    std::set<int> cutPoints;
    while (cutPoints.size() < 4) {
        cutPoints.insert(dist(gen));
    }

    // Convert set to vector and sort it for easy iteration
    std::vector<int> sortedCutPoints(cutPoints.begin(), cutPoints.end());
    sort(sortedCutPoints.begin(), sortedCutPoints.end());

    // Perform 4-point crossover
    int currentCutIndex = 0;
    bool takeFromFirstParent = true;

    for (int i = 0; i < genes; ++i)
    {
        // Check if we passed a cut point
        if (currentCutIndex < sortedCutPoints.size() && i >= sortedCutPoints[currentCutIndex]) {
            takeFromFirstParent = !takeFromFirstParent; // Switch source parent
            currentCutIndex++; // Move to the next cut point
        }

        // Assign gene from the appropriate parent
        child[i] = takeFromFirstParent ? indv1[i] : indv2[i];
    }
}

// void crossover(const std::vector<bool> &indv1, const std::vector<bool> &indv2, std::vector<bool> &child, int genes)
// {
//     std::mt19937 gen(std::random_device{}());
//     std::uniform_int_distribution<int> dist(1, genes - 2); // Ensure cutting points are within bounds

//     int cut1 = dist(gen);
//     int cut2 = dist(gen);
//     int cut3 = dist(gen);

//     // Ensure the cutting points are in ascending order
//     if (cut1 > cut2)
//         std::swap(cut1, cut2);
//     if (cut2 > cut3)
//         std::swap(cut2, cut3);
//     if (cut1 > cut2)
//         std::swap(cut1, cut2);

//     // Perform crossover using the three cutting points
//     for (int i = 0; i < genes; ++i)
//     {
//         if (i < cut1 || (i >= cut2 && i < cut3))
//             child[i] = indv1[i];
//         else
//             child[i] = indv2[i];
//     }
// }

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

double calculateDiversity(const std::vector<std::vector<bool>>& population) {
    // Example: Measure diversity based on the variance of the gene pool
    double mean = 0.0;
    double variance = 0.0;
    int geneLength = population[0].size();

    for (int gene = 0; gene < geneLength; ++gene) {
        int sum = 0;
        for (const auto& individual : population) {
            sum += individual[gene];
        }
        double p = static_cast<double>(sum) / population.size();
        mean += p;
        variance += p * (1 - p);
    }

    mean /= geneLength;
    variance /= geneLength;

    return variance; // Higher variance means higher diversity
}

std::pair<std::vector<int>, std::chrono::duration<double>> genetic()
{
    size_t bitStringLength = std::ceil(std::log2((1.0 * nodes / EPSILON))); // this is the number of bits for one node

    std::vector<int> results;
    std::mutex results_mutex; // Mutex to control access to the results vector
    auto start_time = std::chrono::high_resolution_clock::now();

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
            int mutationRateIncresing = 1;
            for (int i = 0; i < GEN_MAX_NO; ++i)
            {
                std::vector<individ> fitnesses(POPULATION_SIZE);
                calculateFitnessForIndivids(fitnesses, population, bitStringLength, best_function_response);
                std::sort(fitnesses.begin(), fitnesses.end(), compareFitness);
                for (int ii = 0; ii < fitnesses.size(); ++ii)
                    fitnesses[ii].rank = ii + 1;
                std::vector<std::vector<bool>> newPopulation;

                double diversity = calculateDiversity(population);
                const double diversityThreshold = 0.1;
                if (diversity < diversityThreshold) // the calculation is wrong so lets eliminate it
                    mutationRateIncresing = 1;
                else
                    mutationRateIncresing = 1;
                int generatedPopCount = POPULATION_SIZE * 0.15;
                int elitismCount = POPULATION_SIZE * 0.15;
                for (int ii = 0; ii < elitismCount; ++ii)
                {
                    std::vector<bool> v;
                    for (int jj = 0; jj < nodes * bitStringLength; ++jj)
                        v.push_back((*(fitnesses[ii].genes))[jj]);
                    newPopulation.push_back(v);
                }
                while (generatedPopCount < POPULATION_SIZE * 0.5) // 0.5 very good 0.4 seems even better
                {
                    int indv1 = tournamentSelection(fitnesses, 10); // Example tournament size of 5

                    std::vector<bool> child(bitStringLength * nodes);
                    for (int jj = 0; jj < bitStringLength * nodes; ++jj)
                        child[jj] = population[indv1][jj];

                    newPopulation.push_back(child);
                    generatedPopCount++;
                }
                while (generatedPopCount < POPULATION_SIZE)
                {
                    // int indv1 = rouletteWheelSelection(fitnesses);
                    // int indv2 = rouletteWheelSelection(fitnesses);
                    
                    std::uniform_int_distribution<> dist(0, fitnesses.size() - 1);
                    int indv1 = dist(gen);
                    int indv2 = dist(gen);
                    if (indv1 == indv2)
                        continue;

                    std::vector<bool> child(bitStringLength * nodes);
                    crossover(population[indv1], population[indv2], child, bitStringLength * nodes);

                    newPopulation.push_back(child);
                    generatedPopCount++;
                }
                for (int ii = elitismCount; ii < POPULATION_SIZE; ++ii)
                    for (int jj = 0; jj < bitStringLength * nodes; ++jj)
                    {
                        std::uniform_real_distribution<double> dis(0.0, 1.0 - 0.0001);
                        double randomNumber = dis(gen);
                        // if (randomNumber < 20 / (i * nodes * bitStringLength))
                        if (randomNumber < 1.2 * mutationRateIncresing / (nodes * bitStringLength))
                            newPopulation[ii][jj] = !newPopulation[ii][jj];
                    }
                population = std::move(newPopulation);
            }
            // std::cout << "Best function response before hill climbing " << best_function_response << '\n';
            for (auto it : population)
                hill_climb_algorithm(bitStringLength, it, best_function_response);
            // std::cout << "Best function response after hill climbing " << best_function_response << '\n';
            results.push_back(best_function_response);
        }
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time);

    return {results, time_span}; // else this is not the paralele time execution but the total CPU time
}

int main(int argc, char *argv[])
{
    std::srand(std::time(0));

    // first lets read the graph
    std::ifstream in("test.col");
    in >> nodes >> edges;
    for (int i = 1; i <= edges; ++i) {
        int a, b;
        in >> a >> b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }

    auto res = genetic();
    std::cout << "the results are\n";
    double avg = 0;
    for (auto it : res.first)
        std::cout << it << '\n', avg += it;
    std::cout << "THE average is : " << avg / res.first.size() << '\n';
    return 0;
}