#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
#include <ctime>
#include <string>
#include <queue>
#include <fstream>
#include <sstream>
#include <iterator>
#include <cstdio>
#include <Dense>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <unordered_set>

using Eigen::MatrixXd;
using Eigen::MatrixXd;
using namespace std;
using std::milli;
using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::sort;

void print(std::vector<float> const &input)
{
    for (int i = 0; i < input.size(); i++)
    {
        std::cout << input[i] << ' ';
    }
}

void print(std::vector<int> const &input)
{
    for (int i = 0; i < input.size(); i++)
    {
        std::cout << input[i] << ' ';
    }
}

float random_split(vector<float> values, int type)
{

    float split;
    auto result = std::minmax_element(std::begin(values), std::end(values));

    srand(time(NULL));
    std::random_device rd;
    std::mt19937_64 generator(rd());
    if (type == 0)
    {
        // std::normal_distribution<double> distribution(mean, stdev);

        // // cout << mean << " " << stdev << endl;
        // do
        // {
        //     split = distribution(generator);
        // } while (split < *result.first || split > *result.second);

        std::uniform_real_distribution<float> distribution(*result.first, *result.second);
        split = distribution(generator);
        // split = *result.second + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(*result.first-*result.second))); //distribution(generator);
    }
    else
    {   
        std::uniform_int_distribution<int> dist(0, values.size() - 1);
        split = values[dist(generator)];
    }

    // cout << split << endl;
    return split;
}

vector<float> getColumn(const vector<vector<float>> &v, int attribute)
{
    vector<float> col;
    for (auto &row : v)
    {
        col.push_back(row[attribute]);
    }
    return (col);
}

vector<vector<float>> getRows(const vector<vector<float>> &v, vector<int> rows)
{
    vector<vector<float>> out;
    for (int i : rows)
    {
        out.push_back(v[i]);
    }
    return (out);
}

int performSplit(const vector<vector<float>> &dataset, vector<int> &attributesIndices,
                 const vector<int> attributes, const vector<int> &coltypes, vector<int> &left, vector<int> &right)
{
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<int> dist (0,attributesIndices.size()-1);
    int randIndex = dist(mt);
    int attributeIndex = attributesIndices[randIndex];
    *attributesIndices.erase(attributesIndices.begin() + randIndex);
    int attribute = attributes[attributeIndex];
    int type = coltypes[attribute];
    vector<float> data;
    vector<vector<float>> localDataset;

    data = getColumn(dataset, attribute);

    std::unordered_set<float> set;

    for (const float &i : data)
    {
        set.insert(i);
    }

    if (set.size() <= 1)
    {
        return 1;
    }

    float value = random_split(data, type);

    for (int i = 0; i < data.size(); i++)
    {
        if (data[i] < value)
        {
            left.push_back(i);
        }
        else
        {
            right.push_back(i);
        }
    }

    return 0;
}

int performSplit(const vector<vector<float>> &dataset, vector<int> &attributesIndices,
                 const vector<int> attributes, const vector<int> &coltypes, vector<int> &left, vector<int> &right, const vector<int> &nodeIndices)
{
    std::random_device rd;
    std::mt19937_64 mt(rd());
    std::uniform_int_distribution<int> dist (0,attributesIndices.size()-1);
    int randIndex = dist(mt);
    int attributeIndex = attributesIndices[randIndex];
    *attributesIndices.erase(attributesIndices.begin() + randIndex);
    int attribute = attributes[attributeIndex];
    int type = coltypes[attribute];
    vector<float> data;
    vector<vector<float>> localDataset;

    // print(nodeIndices);
    vector<vector<float>> rows = getRows(dataset, nodeIndices);

    data = getColumn(rows, attribute);

    std::unordered_set<float> set;
    for (const float &i : data)
    {
        set.insert(i);
    }

    if (set.size() <= 1)
    {
        return 1;
    }
    float value = random_split(data, type);

    for (int i = 0; i < data.size(); i++)
    {
        if (data[i] < value)
        {
            left.push_back(nodeIndices[i]);
        }
        else
        {
            right.push_back(nodeIndices[i]);
        }
    }
    return 0;
}

struct Node
{
    vector<int> indices;
    // vector<int> instances;
};

MatrixXd build_randomized_tree_and_get_sim(const vector<vector<float>> &data,
                                                        const int &nmin, const vector<int> &coltypes, int nTrees)
{
    int nrows = data.size();
    int ncols = data.front().size();
    MatrixXd matrix = MatrixXd::Zero(nrows, nrows);
    #pragma omp parallel for
    for (int loop = 0; loop < nTrees; loop++)
    {
        queue<Node> nodes;

        vector<int> attributes, attributes_indices;
        for (int i = 0; i < ncols; i++)
        {
            attributes_indices.push_back(i);
            attributes.push_back(i);
        }
        vector<int> left_indices, right_indices;
        performSplit(data, attributes_indices, attributes, coltypes, left_indices, right_indices);

        if (left_indices.size() < nmin)
        {
            for (int instance1 : left_indices)
            {
                for (int instance2 : left_indices)
                {
                    #pragma omp atomic
                    matrix(instance1, instance2) += 1.0/nTrees;
                }
            }
        }

        else
        {
            Node currentNode = {left_indices};
            nodes.push(currentNode);
        }

        if (right_indices.size() < nmin)
        {
            for (int instance1 : right_indices)
            {

                for (int instance2 : right_indices)
                {
                    #pragma omp atomic
                    matrix(instance1, instance2) += 1.0/nTrees;
                }
            }
        }
        else
        {
            Node currentNode = {right_indices};
            nodes.push(currentNode);
        }

        // if (left_indices.size() + right_indices.size() != 150)
        // {
        //     cout << "Error" << endl;
        // }

        // Root node successfully has two children. Now, we iterate over these children.
        while (!nodes.empty())
        {
            if (attributes_indices.size() < 1)
            {
                while (!nodes.empty())
                {
                    vector<int> instances = nodes.front().indices;
                    nodes.pop();
                    for (int instance1 : instances)
                    {
                        for (int instance2 : instances)
                        {
                            #pragma omp atomic
                            matrix(instance1, instance2) += 1.0/nTrees;
                        }
                    }
                }
                break;
            }

            Node currentNode = nodes.front();
            nodes.pop();

            vector<int> nodeIndices = currentNode.indices;

            vector<int> left_indices, right_indices;
            int colNum = performSplit(data, attributes_indices, attributes, coltypes, left_indices, right_indices, nodeIndices);
            if (colNum == 1) // We have a column with only one unique value
            { 
                for (int instance1 : nodeIndices)
                {
                    for (int instance2 : nodeIndices)
                    {
                        #pragma omp atomic
                        matrix(instance1, instance2) += 1.0/nTrees;
                    }
                }
                continue;
            }

            if (left_indices.size() < nmin)
            {
                for (int instance1 : left_indices)
                {
                    for (int instance2 : left_indices)
                    {
                        #pragma omp atomic
                        matrix(instance1, instance2) += 1.0/nTrees;
                    }
                }
            }

            else
            {
                Node currentNode = {left_indices};
                nodes.push(currentNode);
            }

            if (right_indices.size() < nmin)
            {
                for (int instance1 : right_indices)
                {
                    for (int instance2 : right_indices)
                    {
                        #pragma omp atomic
                        matrix(instance1, instance2) += 1.0/nTrees;
                    }
                }
            }

            else
            {
                Node currentNode = {right_indices};
                nodes.push(currentNode);
            }
        

        }
    }
    return matrix;
}

vector<vector<float>> readCSV(string filename, char sep)
{
    ifstream dataFile;
    dataFile.open(filename);
    vector<vector<float>> csv;
    while (!dataFile.eof())
    {
        string line;
        getline(dataFile, line, '\n');
        stringstream buffer(line);
        string tmp;
        vector<float> values;

        while (getline(buffer, tmp, sep))
        {
            values.push_back(strtod(tmp.c_str(), 0));
        }
        csv.push_back(values);
    }

    return csv;
}

int main()
{

    vector<vector<float>> data;
    int nTrees = 250;

    vector<vector<float>> j = readCSV("./data/iris.csv", ',');
    vector<int> labels;
    int nrows = j.size();

    int nmin = floor(nrows / 3);
    cout << nmin << endl;

    data = j;
    vector<int> coltypes;
    for (int i = 0; i < data[0].size(); i++)
    {
        coltypes.push_back(0);
    }

    const auto startTime = high_resolution_clock::now();

    MatrixXd matrix = build_randomized_tree_and_get_sim(data, nmin, coltypes, nTrees);
    const auto endTime = high_resolution_clock::now();

    printf("Time: %fms\n", duration_cast<duration<double, milli>>(endTime - startTime).count());

    // MatrixXd matrix2 = matrix.cast<double>() / nTrees;

    cout << matrix(nrows - 1, nrows - 1) << endl;

    ofstream fichier("./matrix.csv", ios::out | ios::trunc);
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < nrows; j++)
        {
            fichier << matrix(i, j) << '\t';
        }
        fichier << '\n';
    }
    fichier.close();
    // writeToCSVfile("test.csv", matrix);
    return 1;
}
