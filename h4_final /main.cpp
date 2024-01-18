#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <ctime>
#include <algorithm>
#include <queue>
#include <chrono>

#define POPULATION_SIZE 100
#define SAMPLE_SIZE 3
using namespace std;
double INITIAL_TEMPERATURE = 100.0;

int n;  
int **adj; 

void read(string name)
{
    fstream f;
    f.open(name, ios::in);
    int a, b;
    if (n)
    {
        // cleanup
        for (int i = 0; i < n; ++i)
            delete[] adj[i];
        delete[] adj;
    }
    f >> n;
    adj = new int *[n];
    for (int i = 0; i < n; i++)
    {
        adj[i] = new int[n];
        for (int j = 0; j < n; j++)
            adj[i][j] = 0;
    }
    while (!f.eof())
    {
        f >> a >> b;
        a -= 1;
        b -= 1;
        adj[a][b] = 1;
        adj[b][a] = 1;
    }
    f.close();
}

int fittest(vector<int> &chromosome)
{
    int penalty = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (adj[i][j] == 1)
                if (chromosome[i] == chromosome[j])
                    penalty++;
    return penalty;
}

int maxDegree()
{
    int tmp = 0;
    int maxx = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            if (adj[i][j] == 1)
                tmp++;
        maxx = max(maxx, tmp);
        tmp = 0;
    }
    return maxx;
}

void generatePopulation(int maxDegree, vector<pair<vector<int>, int>> &res)
{
    for (int i = 0; i < (POPULATION_SIZE - SAMPLE_SIZE); i++)
    {
        pair<vector<int>, int> temp;
        for (int j = 0; j < n; j++)
        {
            int a = rand() % maxDegree + 1;
            temp.first.push_back(a);
        }
        res.push_back(temp);
    }
}

bool comp(pair<vector<int>, int> &a, pair<vector<int>, int> &b)
{
    return a.second < b.second;
}

void mutate(vector<int> &chromosome, int maxColor, int a)
{
    vector<int> tabu;
    for (int i = 0; i < n; i++)
    {
        if (adj[a][i] == 1)
        {
            tabu.push_back(chromosome[i]);
        }
    }
    int newColor = 1;
    while (find(tabu.begin(), tabu.end(), newColor) != tabu.end())
        newColor++;
    if (newColor >= maxColor)
        newColor = rand() % (maxColor - 1) + 1;
    chromosome[a] = newColor;
}

int colorCount(vector<int> &chromosome)
{
    int res = 0;
    for (int &gene : chromosome)
        res = max(res, gene);
    return res;
}

int colorCount(vector<pair<vector<int>, int>> &population)
{
    int res = 0;
    for (pair<vector<int>, int> &chromosome : population)
        res = max(res, colorCount(chromosome.first));
    return res;
}

void minimalizeColors(vector<int> &chromosome, int maxColors)
{
    vector<int> colors(maxColors);
    for (int gene : chromosome)
        colors[gene - 1]++;
    vector<int> swapTab(maxColors);
    int lowest = 0;
    for (int i = 0; i < colors.size(); i++)
        if (colors.at(i) == 0)
            swapTab.at(i) = -1;
        else
            swapTab.at(i) = lowest++;
    vector<int> newChromosome;
    for (int i : chromosome)
        newChromosome.push_back(swapTab[i - 1] + 1);
    chromosome = std::move(newChromosome);
}

void mate(vector<int> &mother, vector<int> &father, int maxColors, vector<int> &res)
{
    vector<int> toMutate;
    for (int i = 0; i < mother.size(); i++)
    {
        int a = rand() % 100;
        if (i < mother.size() / 3)
            res.push_back(mother[i]);
        else
            res.push_back(father[i]);
        if (a < 20)
        {
            res[res.size() - 1] = -1;
            toMutate.push_back(i);
        }
    }
    for (auto gene : toMutate)
        mutate(res, maxColors, gene);
}

void newPopVol2(vector<pair<vector<int>, int>> &population, int maxColors)
{
    vector<pair<vector<int>, int>> newPopulation;
    int i = 0;
    for (; i < population.size() / 10; i++)
        newPopulation.push_back(population[i]);
    for (; i < population.size(); i++)
    {
        unsigned long long mother = rand() % (population.size() / 2);
        unsigned long long father = rand() % (population.size() / 2);
        for (int k = 0; k < 4; ++k)
        {
            mother = min(mother, 1LL * rand() % (population.size() / 2));
            father = min(father, 1LL * rand() % (population.size() / 2));
        }
        while (father == mother)
            father = (father + 1) % (population.size() / 2);
        pair<vector<int>, int> p;
        mate(population[mother].first, population[father].first, maxColors, p.first);
        p.second = fittest(p.first);
        newPopulation.push_back(p);
    }
    population = std::move(newPopulation);
}

void devaluate(vector<pair<vector<int>, int>> &population, int maxColors)
{
    vector<pair<vector<int>, int>> newPopulation;
    for (pair<vector<int>, int> &p : population)
    {
        vector<int> newChromosome;
        for (int gene : p.first)
            if (gene == maxColors - 1)
                newChromosome.push_back(gene - 1);
            else
                newChromosome.push_back(gene);
        pair<vector<int>, int> pr;
        pr.first = newChromosome;
        pr.second = fittest(newChromosome);
        newPopulation.push_back(pr);
    }
    population = std::move(newPopulation);
}

void mutateWithAnnealing(vector<int> &chromosome, int maxColor, int a, double temperature, vector<int> &res)
{
    vector<int> tabu;
    for (int i = 0; i < n; i++)
        if (adj[a][i] == 1)
            tabu.push_back(chromosome[i]);

    int newColor;
    int count = 0;
    do
    {
        newColor = (rand() % (maxColor)) + 1;
        count++;
    } while (find(tabu.begin(), tabu.end(), newColor) != tabu.end() && count < 1000);

    res[a] = newColor;
}
double acceptanceProbability(int oldFitness, int newFitness, double temperature)
{
    if (newFitness < oldFitness)
        return 1.0;
    return exp((oldFitness - newFitness) / temperature);
}

int geneticAlg(vector<pair<vector<int>, int>> &sample)
{
    int colors = 0;
    int mDeg;
    mDeg = colorCount(sample);
    vector<pair<vector<int>, int>> population;
    vector<pair<vector<int>, int>> newPopulation;
    generatePopulation(mDeg - 1, population);
    colors = colorCount(population);
    for (pair<vector<int>, int> s : sample)
        population.push_back(s);
    sort(population.begin(), population.end(), comp);
    int t = 0;
    int best = mDeg;
    int generationsFromLastDecrese = 0;
    while (t < 4000)
    {
        t++;
        for (auto &indv : population)
            indv.second = fittest(indv.first);
        sort(population.begin(), population.end(), comp);
        newPopVol2(population, colors);
        colors = colorCount(population);
        for (auto &i : population)
            minimalizeColors(i.first, colors);
        colors = colorCount(population);
        sort(population.begin(), population.end(), comp);
        if (population[0].second == 0)
        {
            if (colors < best)
                best = colors;
            devaluate(population, best - 1);
            colors--;
        }
    }
    if (population[0].second != 0)
        for (auto &individual : population)
        {
            int notImprovedIterations = 0;
            double temperature = INITIAL_TEMPERATURE;
            while (notImprovedIterations < 250)
            {
                int oldFitness = individual.second;
                vector<int> temp(individual.first.begin(), individual.first.end());
                mutateWithAnnealing(individual.first, colors, rand() % n, temperature, temp);
                int newFitness = fittest(individual.first);
                if (oldFitness > newFitness)
                {
                    individual.second = newFitness;
                    individual.first = std::move(temp);
                    if (individual.second == 0)
                    {
                        best = min(best, colorCount(individual.first));
                        break;
                    }
                    notImprovedIterations = 0;
                }
                else if (acceptanceProbability(oldFitness, newFitness, temperature) > ((double)rand() / RAND_MAX))
                {
                    individual.second = newFitness;
                    individual.first = std::move(temp);
                    if (individual.second == 0)
                    {
                        best = min(best, colorCount(individual.first));
                        break;
                    }
                }
                notImprovedIterations++;
                temperature *= 0.995;
            }
        }

    return best;
}

void greedy_matrix_arbitrary_vertex(int u, vector<int> &result)
{
    int color;
    auto *available = new vector<bool>;
    queue<int> q;
    for (int i = 0; i < n; i++)
    {
        available->push_back(false);
        result.push_back(-1);
    }
    available->push_back(false);
    q.push(u);
    result[u] = 1;
    while (!q.empty())
    {
        while (!q.empty())
        {
            u = q.front();
            q.pop();
            for (int j = 0; j < n; j++)
                if (adj[u][j] == 1)
                {
                    if (result[j] == -1)
                        q.push(j);
                    else
                        available->at(result[j]) = true;
                }
            for (color = 1; color <= n; color++)
                if (!available->at(color))
                    break;
            result[u] = color;
            for (int j = 0; j < n; j++)
                available->at(j) = false;
        }
        for (int i = 0; i < n; i++)
            if (result[i] == -1)
            {
                q.push(i);
                break;
            }
    }
    delete available;
}

void generateSample(vector<pair<vector<int>, int>> &samplePopulation)
{
    for (int i = 0; i < SAMPLE_SIZE; i++)
    {
        pair<vector<int>, int> samplePair;
        greedy_matrix_arbitrary_vertex(i, samplePair.first);
        samplePair.second = fittest(samplePair.first);
        samplePopulation.push_back(samplePair);
    }
}

int main()
{
    srand(time(NULL));
    int j = 1;
    for (string i = "1"; j <= 9; ++j, i[0]++)
    {
        string f_name = "./tests/test_" + i;
        std::cout << f_name << '\n';
        read(f_name);
        vector<pair<vector<int>, int>> samplePopulation;
        generateSample(samplePopulation);
        cout << "Final result: " << geneticAlg(samplePopulation) << endl;
    }
}