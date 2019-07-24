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
#include <nlohmann/json.hpp>
#include <cstdio>
#include <iomanip>
using json = nlohmann::json;
using namespace std;
using std::vector;

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

void print(std::vector<int> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input.at(i) << ' ';
	}
}

vector<double> getColumn(vector<vector<double>> &v, int attribute) {
    vector<double> col;
    for(auto& row:v) {
        col.push_back(row.at(attribute));
        }
    return(col);
    }

vector<vector<double>> getRows(vector<vector<double>>&v, vector<int> rows) {
    vector<vector<double>>out;
    for(int i:rows) {
            out.push_back(v.at(i));
        }
    return(out);
    }



void performSplit(vector<vector<double>> dataset, vector<int>& attributesIndices, 
vector<int> attributes, vector<int> &left, vector<int> &right, vector<int> nodeIndices={}) {
    // cout << attributesIndices.size() << endl;
    int randIndex = rand() % attributesIndices.size();
    // cout << "a" << endl;
    int attributeIndex = attributesIndices.at(randIndex);
    // cout << "b" << endl;
    *attributesIndices.erase(attributesIndices.begin()+randIndex); 
    int attribute = attributes.at(attributeIndex);
    // cout << "c" << endl;
    vector<double> data;
    vector<vector<double>> localDataset;
    if (nodeIndices.size() == 0) {
        // cout << "d1" << endl;

        data = getColumn(dataset, attribute);
        // cout << "d2" << endl;
    }
    else {
        // cout << "e1" << endl;

        vector<vector<double>> rows = getRows(dataset, nodeIndices);
        // cout << "e2" << endl;

        data = getColumn(rows, attribute);
        // getColumn(dataset, attribute, data);
        // cout << "f" << endl;


    }
    double value = random_split(data, 0);
    // cout << "g" << endl;

    for(int i = 0; i < data.size(); i++) {
        if(data.at(i) < value) {
            if (nodeIndices.size() == 0) {
                left.push_back(i);
            }
            else {
                left.push_back(nodeIndices.at(i));
            }

        }
        else {
            if (nodeIndices.size() == 0) {
                right.push_back(i);
            }
            else {
                right.push_back(nodeIndices.at(i));
            }
        }
    }
}

struct Node {
    vector<int> indices;
    vector<int> instances;
};

extern "C++"
vector<vector<double>> build_randomized_tree_and_get_sim(vector<vector<double>> data, 
double nmin, vector<int> coltypes) {


    srand(time(NULL));
    int nrows = data.size();
    int ncols = data.front().size();
    // cout << nrows << "\n";

    vector<vector<double>> matrix(nrows, vector<double>(nrows, 0)); 
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

    }
    for (int j : left_instances) {
        left_data.push_back(data.at(j));
    }   
    for (int i=0; i < right_indices.size(); i++) {
        int index = right_indices.at(i);
        right_instances.push_back(instancesList.at(index));

    }
    for (int j:right_instances) {
        right_data.push_back(data.at(j));
    }

    if (left_indices.size() < nmin) {
        for (int instance1:left_instances) {
            for (int instance2:left_instances) {
                matrix.at(instance1).at(instance2) += 1;
                if (instance1 != instance2) {
                    matrix.at(instance2).at(instance1) += 1;
                }
            } 
        }
    }


    else {
        Node currentNode = {left_indices, left_instances};
        nodes.push(currentNode);
    }

    if(right_indices.size() < nmin) {
        for (int instance1:right_instances) {
            for (int instance2:right_instances) {
                matrix.at(instance1).at(instance2) += 1;
                if (instance1 != instance2) {
                    matrix.at(instance2).at(instance1) += 1;
                }            
            }
        }
    }
    else {
        Node currentNode = {right_indices, right_instances};
        nodes.push(currentNode);
    }

            for (int const instance1:left_instances) {
                for (int const instance2: right_instances) {
                    if (instance1 == instance2) {
                        cout << "Même instance dans deux branches différentes ici !" << endl;
                    }
                }
            }

    // Root node successfully has two children. Now, we iterate over these children.

    while (!nodes.empty()) {
        if (attributes_indices.size() < 1) {
            while (!nodes.empty()) {
                vector<int> instances = nodes.front().instances;
                nodes.pop();
                for (int instance1:instances) {
                    for (int instance2:instances) {
                        matrix.at(instance1).at(instance2) += 1;
                        if (instance1 != instance2) {
                            matrix.at(instance2).at(instance1) += 1;
                        }                       
                    }
                }

            }
        break;

        }

        Node currentNode = nodes.front();
        nodes.pop();

        vector<double> col;
        vector<int> nodeIndices = currentNode.indices;
        vector<int> left_indices, right_indices;

        if (data.size() > 1) {
            performSplit(data, attributes_indices, attributes, left_indices, right_indices, nodeIndices);
            vector<int> left_instances, right_instances;
            vector<vector<double>> left_data, right_data;

            for (int i=0; i < left_indices.size(); i++) {
                int index = left_indices.at(i);
                left_instances.push_back(instancesList.at(index));
            }

            for (int j : left_instances) {
                left_data.push_back(data.at(j));
            }  
 
            for (int i=0; i < right_indices.size(); i++) {
                int index = right_indices.at(i);
                right_instances.push_back(instancesList.at(index));
            }

            for (int j:right_instances) {
                right_data.push_back(data.at(j));
            }
            if (left_indices.size() < nmin) {
                for (int instance1:left_instances) {
                    for (int instance2:left_instances) {
                        matrix.at(instance1).at(instance2) += 1;
                        if (instance1 != instance2) {
                            matrix.at(instance2).at(instance1) += 1;
                        }                
                    }
                }
            }

            else {
                Node currentNode = {left_indices, left_instances};
                nodes.push(currentNode);

            }


            if(right_indices.size() < nmin) {
                for (int instance1:right_instances) {
                    for (int instance2:right_instances) {
                        matrix.at(instance1).at(instance2) += 1;
                        if (instance1 != instance2) {
                            matrix.at(instance2).at(instance1) += 1;
                        }
                    }
                }

            }

            else {
                Node currentNode = {right_indices, right_instances};
                nodes.push(currentNode);
            }

        }
        else {
            cout << "Else" << "\n";
        }

        
    }
    return matrix;
}
int main() {
    // vector<double> row1 = {1.3, 0.1, 2.3, 7.2, 8.4};
    // vector<double> row2 = {8.3, 4.1, 1.3, 3.2, 9.5};
    // vector<double> row3 = {1.0, 2.0, 3.0, 4.0, 5.0};
    // vector<double> row4 = {0.1, 0.2, 0.3, 0.4, 0.5};
    // vector<vector<double>> data  = {row1, row2, row3, row4}; 
    vector<vector<double>> data;
    int nmin = 3;
    // cout << data.size();
    std::ifstream i("pima.json");
    json j;
    i >> j;
    // int n_rows = j;
    // int n_cols = j.at(0).size();
    // cout << n_cols + " " + n_rows << "\n";
    for (auto& element : j) {
        vector<double> row;
        for(auto& element2 : element) {
            row.push_back(element2);
            // std::cout << element2 << '\n';
        }
        data.push_back(row);
    }
    vector<int> coltypes;
    for (int i=0; i < data.at(0).size(); i++) {
        coltypes.push_back(1);
    }
    vector<vector<double>> matrix = build_randomized_tree_and_get_sim(data, nmin, coltypes);
    int errors = 0;
    for (int i=0; i < matrix.size(); i++) {
        // for (int j=0; j < matrix.size(); j++) {
            // cout << matrix[i][i] << endl;
            if (matrix[i][i] == 0) {
                errors++;
            // }
        }
    }
cout << errors << "\n";

    return 1;
}

