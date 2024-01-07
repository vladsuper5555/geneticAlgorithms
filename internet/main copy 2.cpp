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
#define MUTATION_PERCENT 25

using namespace std;

int n;     // number of vertices in graph
int **adj; // matrix representing graph

void show()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%d ", adj[i][j]);
        printf("\n");
    }
}

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

int fittest(const int *chromosome)
{
    int penalty = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (adj[i][j] == 1)
            {
                if (chromosome[i] == chromosome[j])
                {
                    penalty++;
                }
            }
        }
    }
    return penalty;
}

int fittest(vector<int> *chromosome)
{
    int penalty = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (adj[i][j] == 1)
            {
                if (chromosome->at(i) == chromosome->at(j))
                {
                    penalty++;
                }
            }
        }
    }
    return penalty;
}

int maxDegree()
{
    int tmp = 0;
    int maks = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (adj[i][j] == 1)
            {
                tmp++;
            }
        }
        maks = max(maks, tmp);
        tmp = 0;
    }
    return maks;
}

vector<pair<vector<int> *, int> *> *generatePopulation(int maxDegree)
{
    auto *res = new vector<pair<vector<int> *, int> *>;
    for (int i = 0; i < (POPULATION_SIZE - SAMPLE_SIZE); i++)
    {
        auto *tmp = new vector<int>;
        for (int j = 0; j < n; j++)
        {
            int a = rand() % maxDegree + 1;
            tmp->push_back(a);
        }
        auto *p = new pair<vector<int> *, int>;
        *p = make_pair(tmp, fittest(tmp));
        res->push_back(p);
    }
    return res;
}

vector<vector<int> *> *crossover(vector<int> *first, vector<int> *second)
{
    int a = rand() % (n - 1);
    auto *newFirst = new vector<int>;
    auto *newSecond = new vector<int>;
    int i = 0;
    for (i; i < a; i++)
    {
        newFirst->push_back(second->at(i));
        newSecond->push_back(first->at(i));
    }
    for (i; i < n; i++)
    {
        newFirst->push_back(first->at(i));
        newSecond->push_back(second->at(i));
    }
    auto *res = new vector<vector<int> *>;
    res->push_back(newFirst);
    res->push_back(newSecond);
    delete newFirst;
    delete newSecond;
    return res;
}

bool comp(pair<vector<int> *, int> *a, pair<vector<int> *, int> *b)
{
    return a->second < b->second;
}

void mutate(vector<int> *chromosome, int maxColor, int a)
{
    vector<int> tabu;
    for (int i = 0; i < n; i++)
    {
        if (adj[a][i] == 1)
        {
            tabu.push_back(chromosome->at(i));
        }
    }
    int newColor = 1;
    while (find(tabu.begin(), tabu.end(), newColor) != tabu.end())
        newColor++;
    if (newColor >= maxColor)
        newColor = rand() % (maxColor - 1) + 1;
    chromosome->at(a) = newColor;
}

int colorCount(vector<int> *chromosome)
{
    int res = 0;
    for (int gene : *chromosome)
        res = max(res, gene);
    return res;
}

int colorCount(vector<pair<vector<int> *, int> *> *population)
{
    int res = 0;
    for (pair<vector<int> *, int> *chromosome : *population)
        res = max(res, colorCount(chromosome->first));
    return res;
}

vector<int> *minimalizeColors(vector<int> *chromosome, int maxColors)
{
    vector<int> colors(maxColors);
    for (int gene : *chromosome)
        colors.at(gene - 1)++;
    vector<int> swapTab(maxColors);
    int lowest = 0;
    for (int i = 0; i < colors.size(); i++)
        if (colors.at(i) == 0)
            swapTab.at(i) = -1;
        else
            swapTab.at(i) = lowest++;
    auto *newChromosome = new vector<int>;
    for (int i : *chromosome)
        newChromosome->push_back(swapTab.at(i - 1) + 1);
    delete chromosome;
    return newChromosome;
}

vector<int> *mate(vector<int> *mother, vector<int> *father, int maxColors)
{
    auto res = new vector<int>;
    auto toMutate = new vector<int>;
    for (int i = 0; i < mother->size(); i++)
    {
        int a = rand() % 100;
        if (a < 45)
            res->push_back(mother->at(i));
        else if (a < 90)
            res->push_back(father->at(i));
        else
        {
            res->push_back(-1);
            toMutate->push_back(i);
        }
    }
    for (auto gene : *toMutate)
        mutate(res, maxColors, gene);
    return res;
}

vector<pair<vector<int> *, int> *> *newPopVol2(vector<pair<vector<int> *, int> *> *population, int maxColors)
{
    auto *newPopulation = new vector<pair<vector<int> *, int> *>;
    int i = 0;
    for (i; i < population->size() / 10; i++)
    {
        // mutate(population->at(i)->first, maxColors, rand()%n);
        // mutate(population->at(i)->first, maxColors, rand()%n);
        newPopulation->push_back(population->at(i));
    }
    for (i; i < population->size(); i++)
    {
        int mother = rand() % (population->size() / 2);
        int father = rand() % (population->size() / 2);
        while (father == mother)
        {
            father = (father + 1) % (population->size() / 2);
        }
        auto *p = new pair<vector<int> *, int>;
        *p = make_pair(mate(population->at(mother)->first, population->at(father)->first, maxColors), 0);
        p->second = fittest(p->first);
        newPopulation->push_back(p);
    }
    // for (auto &it : *population)
    //     delete it->first, delete it;
    // delete population;
    return newPopulation;
}

vector<pair<vector<int> *, int> *> *devaluate(vector<pair<vector<int> *, int> *> *population, int maxColors)
{
    auto *newPopulation = new vector<pair<vector<int> *, int> *>;
    for (pair<vector<int> *, int> *p : *population)
    {
        auto *newChromosome = new vector<int>;
        for (int gene : *p->first)
            if (gene == maxColors - 1)
                newChromosome->push_back(gene - 1);
            else
                newChromosome->push_back(gene);
        auto *pr = new pair<vector<int> *, int>;
        *pr = make_pair(newChromosome, fittest(newChromosome));
        newPopulation->push_back(pr);
    }
    for (auto it : *population)
        delete it->first, delete it;
    delete population;
    return newPopulation;
}
int geneticAlg(vector<pair<vector<int> *, int> *> *sample)
{
    int colors = 0;
    int mDeg;
    if (sample->empty())
        mDeg = maxDegree();
    else
        mDeg = colorCount(sample);
    vector<pair<vector<int> *, int> *> *population;
    vector<pair<vector<int> *, int> *> *newPopulation;
    population = generatePopulation(mDeg - 1);
    colors = colorCount(population);
    for (pair<vector<int> *, int> *s : *sample)
        population->push_back(s);
    sort(population->begin(), population->end(), comp);
    int t = 0;
    int best = mDeg;
    vector<int> *bestChr = population->at(0)->first;
    auto start = chrono::steady_clock::now();
    while (t < 2000)
    {
        t++;
        colors = colorCount(population);
        newPopulation = newPopVol2(population, colors);
        // not sure it is a good parctice to keep the crossover here continue the testing

        // for (int i = 0; i < POPULATION_SIZE; i += 2)
        // {
        //     auto *tmp = new vector<vector<int> *>;
        //     tmp = crossover((*newPopulation)[i]->first, (*newPopulation)[i + 1]->first);
        //     (*newPopulation)[i]->first = (*tmp)[0];
        //     newPopulation->at(i + 1)->first = tmp->at(1);
        // }
        population = newPopulation;
        colors = colorCount(population);
        for (auto &i : *population)
        {
            vector<int> *tmp = minimalizeColors(i->first, colors);
            i->first = tmp;
        }
        colors = colorCount(population);
        sort(population->begin(), population->end(), comp);
        cout << t << ": " << colors << "(" << population->at(0)->second << ")\t";
        if (population->at(0)->second == 0)
        {
            if (colors < best)
                best = colors;
            population = devaluate(population, best - 1);
            colors--;
            // cout << "decresing the colors \n";
            // colors = colorCount(population);
        }
        // for (int i = 0; i < POPULATION_SIZE; i += 2)
        // {
        //     auto *tmp = new vector<vector<int> *>;
        //     tmp = crossover((*population)[i]->first, (*population)[i + 1]->first);
        //     (*population)[i]->first = (*tmp)[0];
        //     population->at(i + 1)->first = tmp->at(1);
        // }
    }
    return best;
}

vector<int> *greedy_matrix_arbitrary_vertex(int u)
{
    int color;
    auto *available = new vector<bool>;
    auto *result = new vector<int>;
    queue<int> q;
    for (int i = 0; i < n; i++)
    {
        available->push_back(false);
        result->push_back(-1);
    }
    available->push_back(false);
    q.push(u);
    result->at(u) = 1;
    while (!q.empty())
    {
        while (!q.empty())
        {
            u = q.front();
            q.pop();
            for (int j = 0; j < n; j++)
                if (adj[u][j] == 1)
                {
                    if (result->at(j) == -1)
                        q.push(j);
                    else
                        available->at(result->at(j)) = true;
                }
            for (color = 1; color <= n; color++)
                if (!available->at(color))
                    break;
            result->at(u) = color;
            for (int j = 0; j < n; j++)
                available->at(j) = false;
        }
        for (int i = 0; i < n; i++)
            if (result->at(i) == -1)
            {
                q.push(i);
                break;
            }
    }
    delete available;
    return result;
}

vector<pair<vector<int> *, int> *> *generateSample()
{
    auto *samplePopulation = new vector<pair<vector<int> *, int> *>;
    for (int i = 0; i < SAMPLE_SIZE; i++)
    {
        auto *sample = greedy_matrix_arbitrary_vertex(i);
        // cout << "Sample: " << i << endl;
        auto *samplePair = new pair<vector<int> *, int>;
        *samplePair = make_pair(sample, fittest(sample));
        samplePopulation->push_back(samplePair);
    }
    return samplePopulation;
}

int main()
{
    srand(time(NULL));
    int j = 6;
    for (string i = "6"; j <= 6; ++j, i[0]++)
    {
        string f_name = "./tests/test_" + i;
        cout << "The file name is " << f_name << endl;
        read(f_name);
        cout << "The file was read\n";
        auto *samplePopulation = generateSample();
        int max_color = 0;
        for (int i = 0; i < n; i++)
        {
            // cout << samplePopulation->at(0)->first->at(i) << "\t";
            max_color = max(max_color, samplePopulation->at(0)->first->at(i) + 1);
        }
        // cout << endl
        //     << "Penalty: " << samplePopulation->at(0)->second << endl;
        // cout << "Max Degree " << maxDegree() << endl;
        cout << "Final result: " << geneticAlg(samplePopulation) << endl;
    }
}