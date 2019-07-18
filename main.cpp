#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
#include <ctime>
#include <string>
#include <queue>
using namespace std;

double mean(const std::vector<double> &v)
{
    double sum = 0;

    for (auto &each : v)
        sum += each;

    return sum / v.size();
}

double sd(const std::vector<double> &v)
{
    double square_sum_of_difference = 0;
    double mean_var = mean(v);
    auto len = v.size();

    double tmp;
    for (auto &each : v)
    {
        tmp = each - mean_var;
        square_sum_of_difference += tmp * tmp;
    }

    return std::sqrt(square_sum_of_difference / (len - 1));
}
double random_split(vector<double> values, int type)
{

    double split;
    double min, max;
    auto result = std::minmax_element(std::begin(values), std::end(values));
    // srand(time(NULL));
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(*result.first, *result.second);

    double number = distribution(generator);
    if (type == 0)
    {
        split = distribution(generator);
    }
    else
    {
        split = distribution(generator);
    }
    return split;
}

void print(std::vector<double> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input.at(i) << ' ';
	}
}

void getColumn(vector<vector<double>> &v, int attribute, vector<double>&col) {
    for(auto& row:v) {
        col.push_back(row.at(attribute));
        }
    }

void getRows(vector<vector<double>>&v, vector<int> rows, vector<vector<double>>&out) {
    for(vector<double> row:v) {
            out.push_back(row);
        }
    }



void performSplit(vector<vector<double>> dataset, vector<int>& attributesIndices, vector<int> attributes, vector<int> &left, vector<int> &right, vector<int> nodeIndices={}) {
    int randIndex = rand() % attributesIndices.size();
    int attributeIndex = attributesIndices.at(randIndex);
    *attributesIndices.erase(attributesIndices.begin()+randIndex); 
    int attribute = attributes.at(attributeIndex);
    vector<double> data;
    vector<vector<double>> localDataset;
    if (nodeIndices.size() == 0) {
        getColumn(dataset, attribute, data);
    }
    else {
        // getRows(dataset, nodeIndices, localDataset);
        // getColumn(localDataset, attribute, data);
        getColumn(dataset, attribute, data);


    }
    double value = random_split(data, 0);

    for(int i = 0; i < data.size(); i++) {
        if(data.at(i) < value) {
            left.push_back(i);
        }
        else {
            right.push_back(i);
        }
    }
}

struct Node {
    vector<int> indices;
    vector<int> instances;
    vector<vector<double>> data;
};

vector<vector<double>> build_randomized_tree_and_get_sim(vector<vector<double>> data, 
double nmin, vector<int> coltypes) {


    srand(time(NULL));
    int nrows = data.size();
    int ncols = data.front().size();
    cout << nrows << "\n";

    vector<vector<double>> matrix(nrows, vector<double>(ncols, 0)); 
    queue<Node> nodes;

    vector<int> instancesList;
    for (int i =0; i < nrows; i++) {
        instancesList.push_back(i);
    } 
    vector<int> attributes, attributes_indices;
    for (int i = 0; i < ncols; i++) {
        attributes_indices.push_back(i);
        attributes.push_back(i);
    }

    vector<double> col; // Each column contains rowsize number of elements, i.e, the number of instances 
    vector<int> left_indices, right_indices;
    performSplit(data, attributes_indices, attributes, left_indices, right_indices);
    vector<int> left_instances, right_instances;
    vector<vector<double>> left_data, right_data;
    for (int i=0; i < left_indices.size(); i++) {
        int index = left_indices.at(i);
        left_instances.push_back(instancesList.at(index));
        for (int j=0; j < left_instances.size(); j++) {
            left_data.push_back(data.at(j));
        }
    }
    for (int i=0; i < right_indices.size(); i++) {
        int index = right_indices.at(i);
        right_instances.push_back(instancesList.at(index));
        for (int j=0; j < right_instances.size(); j++) {
            right_data.push_back(data.at(j));
        }
    }

    if (left_indices.size() < nmin) {
        for (int instance1:left_instances) {
            for (int instance2:left_instances) {
                matrix.at(instance1).at(instance2) += 1;
            } 
        }
    }
    else {
        Node currentNode = {left_indices, left_instances, left_data};
        nodes.push(currentNode);
    }

    if(right_indices.size() < nmin) {
        for (int instance1:right_instances) {
            for (int instance2:right_instances) {
                matrix.at(instance1).at(instance2) += 1;
            }
        }
    }
    else {
        Node currentNode = {right_indices, right_instances, right_data};
        nodes.push(currentNode);
    }
    while (!nodes.empty()) {
        Node currentNode = nodes.front();
        cout << "1" << "\n";
        nodes.pop();
    }
    print(matrix.at(1));
    return matrix;
}

int main() {
    vector<double> row1 = {1.3, 0.1, 2.3, 7.2};
    vector<double> row2 = {8.3, 4.1, 1.3, 3.2};
    vector<double> row3 = {1.0, 2.0, 3.0, 4.0};
    vector<double> row4 = {0.1, 0.2, 0.3, 0.4};
    vector<vector<double>> data  = {row1, row2, row3, row4}; 

    int nmin = 3;
    vector<int> coltypes{0,0,0,0};
    vector<vector<double>> matrix = build_randomized_tree_and_get_sim(data, nmin, coltypes);
    return 1;
}