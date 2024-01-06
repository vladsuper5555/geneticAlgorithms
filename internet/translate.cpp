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
#define SAMPLE_SIZE 1
#define MUTATION_PERCENT 25

using namespace std;

void translate(string name)
{
    fstream input;
    fstream output;
    string buffer;
    input.open(name+".col.b", ios::in|ios::binary);
    output.open(name+".txt", ios::out);
    while(!input.eof()){
        getline(input, buffer, '\n');
        output << buffer << endl;
    }
    input.close();
    output.close();
}

int main()
{
    srand(time(NULL));
    string f_name = "flat1000_50_0";
    translate(f_name);
}