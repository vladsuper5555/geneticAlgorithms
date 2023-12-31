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
#include <limits.h>
#define DBL_MIN -(1e10)
const int MAX_DATA_GATHERINGS = 40;
const int T_MAX_HILL = 1000;
const double EPSILON = 0.00001;
// pass as reference in the arguments in the functions
double power(double a, int exponent)
{
    double res = 1.0;
    while (exponent)
    {
        if ((exponent & 1))
            res = res * a;
        exponent >>= 1;
        a = a * a;
    }
    return res;
}
// to be remade maybe the values are with exactly one unit under the real one
double Michalewicz(std::vector<double> inputs)
{
    double sum = 0;
    double m = 10;
    for (size_t index = 0; index < inputs.size(); index++)
    {
        double value = inputs[index];
        sum += std::sin(value) * power(std::sin(((index + 1) * value * value) / M_PI), 2 * m);
    }
    return -sum;
}

std::unordered_map<double, double> cache;
double Schewefel(std::vector<double> inputs)
{
    double sum = 0;
    for (double value : inputs)
    {
        auto it = cache.find(value);
        if (it == cache.end())
        {
            double v = -value * std::sin(std::sqrt(std::abs(value)));
            sum += v;
            cache.emplace(value, v);  // Use emplace instead of operator[]
        }
        else
            sum += it->second;  // Directly access the cached value from the iterator
    }
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
    Schewefel,
    Michalewicz,
    Rastrigin,
    De_Jong
};

std::pair<double, double> ranges[] = {
    {-500.0, 500.0},
    {0.0, M_PI},
    {-5.12, 5.12},
    {-5.12, 5.12}
    };

std::string functionNames[] = {
    "Schewefel",
    "Michalewicz",
    "Rastrigin",
    "De_Jong"
    };

struct Function
{
    FunctionPointer func;
    std::string name;
    std::pair<double, double> range;
};

std::vector<double> decodeBitset(bool *bits, const std::pair<double, double> &rangeInterval, size_t bitStringLength, int dimension)
{
    double rangeValue = rangeInterval.second - rangeInterval.first;

    std::vector<double> values;
    const int bitCount = dimension * bitStringLength;
    for (size_t i = 0; i < bitCount; i += bitStringLength)
    {
        double value = 0;
        double range_copy = rangeValue / 2;
        for (int bit = i; bit < i + bitStringLength; ++bit, range_copy /= 2)
            value += range_copy * bits[bit];
        double value_in_range = value + rangeInterval.first;
        values.push_back(value_in_range);
    }
    return values;
}

double chooseNextNeighbour(const Function &function, bool *currentBits, const int &method, double initialFunctionValue, const std::pair<double, double> &rangeInterval, int dimension, size_t bitStringLength)
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
        else if (method == 2 && functionValue < bestValue)
        {
            bestValue = functionValue;

            for (int i = 0; i < dimension * bitStringLength; ++i)
                bestNeighbour[i] = currentBits[i];
            break;
        }
        else if (method == 3 && functionValue > bestValue && functionValue < initialFunctionValue)
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

std::vector<std::pair<double, std::chrono::duration<double>>> hill_climb_algorithm(const Function &function, int dimension, const int method)
{
    std::mt19937_64 gen(static_cast<std::mt19937_64::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    // gen.discard(31337);
    gen.seed(gen());
    std::uniform_int_distribution<> distrib(0, 1);
    size_t bitStringLength = std::ceil(std::log2((function.range.second - function.range.first) / EPSILON));

    std::vector<std::pair<double, std::chrono::duration<double>>> results;
    for (int test = 1; test < MAX_DATA_GATHERINGS; ++test)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        double best_function_response = std::numeric_limits<double>::infinity();
        bool currentBits[bitStringLength * dimension];
        for (int i = 0; i < T_MAX_HILL; ++i)
        {
            gen.seed(gen());
            for (size_t j = 0; j < bitStringLength * dimension; ++j)
                // currentBits[j] = (std::rand() % 2);
                currentBits[j] = distrib(gen) % 2;
            double current_value = function.func(decodeBitset(currentBits, function.range, bitStringLength, dimension));
            bool local = false;

            while (!local)
            {
                auto nextNeighbourValue = chooseNextNeighbour(function, currentBits, method, current_value, function.range, dimension, bitStringLength);

                if (nextNeighbourValue < current_value)
                    current_value = nextNeighbourValue;
                else
                    local = true;
            }

            if (current_value < best_function_response)
            {
                best_function_response = current_value;
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        // std::cout << "Iteration " << test + 1 << ": Time taken = " << time_span.count() << " seconds. New best value = " << best_function_response << '\n';
        results.push_back({best_function_response, time_span});
    }
    return results;
}



int main(int argc, char* argv[])
{
    std::srand(std::time(0));

    std::vector<Function> functions;
    for (size_t i = 0; i < sizeof(functionDefinitions) / sizeof(FunctionPointer); ++i)
    {
        functions.push_back({functionDefinitions[i], functionNames[i], ranges[i]});
    }

    std::vector<int> dimensions = {5, 10, 30};
    int method = 1; // default value
    // 1 is best
    // 2 is first
    // 3 is worst
    if (argc > 1) {
        method = std::stoi(argv[1]);
    }

    std::string filename = "hill_climbing_method" + std::to_string(method) + ".txt";
    std::ofstream output_file(filename);

    for (const auto &function : functions)
    {
        for (const auto &dimension : dimensions)
        {
            auto res = hill_climb_algorithm(function, dimension, method);
            for (auto r : res)
                std::cout << "Function: " << function.name << ", Dimension: " << dimension << ", Min Value: " << std::fixed <<std::setprecision(5) << r.first << ", Time: " << r.second.count() << std::endl;
        }
    }

    output_file.close();
    return 0;
}