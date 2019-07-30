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
#include <Dense>
#include <chrono>
#include <omp.h>
#include <fstream>

using json = nlohmann::json;
using Eigen::MatrixXd;
using namespace std;
using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::milli;

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
    srand(time(NULL));
    std::random_device rd;
    std::default_random_engine generator(rd());

    // double number = distribution(generator);
    if (type == 0) {
        std::uniform_real_distribution<double> distribution(*result.first, *result.second);
        split = distribution(generator);
    }
    else {
        std::uniform_int_distribution<int> dist(0, values.size() - 1);
        split = values[dist(generator)];
    }
    return split;
}

void print(std::vector<double> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input[i] << ' ';
	}
}

void print(std::vector<int> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input[i] << ' ';
	}
}

vector<double> getColumn(const vector<vector<double>> &v, int attribute) {
    vector<double> col;
    for(auto& row:v) {
        col.push_back(row[attribute]);
        }
    return(col);
    }

vector<vector<double>> getRows(const vector<vector<double>> &v, vector<int> rows) {
    vector<vector<double>>out;
    for(int i:rows) {
            out.push_back(v[i]);
        }
    return(out);
    }



void performSplit(const vector<vector<double>>& dataset, vector<int>& attributesIndices, 
vector<int> attributes, const vector<int>& coltypes, vector<int> &left, vector<int> &right, const vector<int> &nodeIndices={}) {
    // cout << attributesIndices.size() << endl;
    int randIndex = rand() % attributesIndices.size();
    // cout << "a" << endl;
    int attributeIndex = attributesIndices[randIndex];
    // cout << "b" << endl;
    *attributesIndices.erase(attributesIndices.begin()+randIndex); 
    int attribute = attributes[attributeIndex];
    int type = coltypes[attribute];
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
    double value = random_split(data, type);
    // cout << "g" << endl;

    for(int i = 0; i < data.size(); i++) {
        if(data[i] < value) {
            if (nodeIndices.size() == 0) {
                left.push_back(i);
            }
            else {
                left.push_back(nodeIndices[i]);
            }

        }
        else {
            if (nodeIndices.size() == 0) {
                right.push_back(i);
            }
            else {
                right.push_back(nodeIndices[i]);
            }
        }
    }
}

struct Node {
    vector<int> indices;
    vector<int> instances;
};

extern "C++"
MatrixXd build_randomized_tree_and_get_sim(const vector<vector<double>>& data, 
const double& nmin, const vector<int>& coltypes) {


    srand(time(NULL));
    int nrows = data.size();
    int ncols = data.front().size();
    // cout << nrows << "\n";

    // vector<vector<double>> matrix(nrows, vector<double>(nrows, 0)); 
    MatrixXd matrix = MatrixXd::Zero(nrows, nrows);
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
    performSplit(data, attributes_indices, attributes, coltypes, left_indices, right_indices);

    vector<int> left_instances, right_instances;
    vector<vector<double>> left_data, right_data;
    for (int i=0; i < left_indices.size(); i++) {
        int index = left_indices[i];
        left_instances.push_back(instancesList[index]);

    }
    for (int j : left_instances) {
        left_data.push_back(data[j]);
    }   
    for (int i=0; i < right_indices.size(); i++) {
        int index = right_indices[i];
        right_instances.push_back(instancesList[index]);

    }
    for (int j:right_instances) {
        right_data.push_back(data[j]);
    }

    if (left_indices.size() < nmin) {
        for (int instance1:left_instances) {
            for (int instance2:left_instances) {
                matrix(instance1, instance2)+= 1.0;
                // if (instance1 != instance2) {
                //     matrix(instance2, instance1)+= 1.0;
                // }
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
                matrix(instance1, instance2)+= 1.0;
                // if (instance1 != instance2) {
                //     matrix(instance2, instance1)+= 1.0;
                // }            
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
                        matrix(instance1, instance2)+= 1.0;
                        // if (instance1 != instance2) {
                        //     matrix(instance2, instance1)+= 1.0;
                        // }                       
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
            performSplit(data, attributes_indices, attributes, coltypes, left_indices, right_indices, nodeIndices);
            vector<int> left_instances, right_instances;
            vector<vector<double>> left_data, right_data;

            for (int i=0; i < left_indices.size(); i++) {
                int index = left_indices[i];
                left_instances.push_back(instancesList[index]);
            }

            for (int j : left_instances) {
                left_data.push_back(data[j]);
            }  
 
            for (int i=0; i < right_indices.size(); i++) {
                int index = right_indices[i];
                right_instances.push_back(instancesList[index]);
            }

            for (int j:right_instances) {
                right_data.push_back(data[j]);
            }
            if (left_indices.size() < nmin) {
                for (int instance1:left_instances) {
                    for (int instance2:left_instances) {
                        matrix(instance1, instance2)+= 1.0;
                        // if (instance1 != instance2) {
                        //     matrix(instance2, instance1)+= 1.0;
                        // }                
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
                        matrix(instance1, instance2)+= 1.0;
                        // if (instance1 != instance2) {
                        //     matrix(instance2, instance1)+= 1.0;
                        // }
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

// void writeToCSVfile(string name, MatrixXd matrix)
// {
//   ofstream file(name.c_str());

//   for(int  i = 0; i < matrix.rows(); i++){
//       for(int j = 0; j < matrix.cols(); j++){
//          string str = lexical_cast<std::string>(matrix(i,j));
//          if(j+1 == matrix.cols()){
//              file<<str;
//          }else{
//              file<<str<<',';
//          }
//       }
//       file<<'\n';
//   }
// }
vector<vector<double>> readCSV(string filename, char sep) {
    ifstream dataFile;
    dataFile.open(filename);
    vector<vector<double>> csv;
    while(!dataFile.eof()) {
        string line;
        getline(dataFile, line, '\r');
        stringstream buffer(line);
        string tmp;
        vector<double> values;

        while(getline(buffer, tmp, '\t') ) {
            values.push_back(strtod(tmp.c_str(), 0));
        }
        csv.push_back(values);
}


    return csv;
}

int main() {
 
    vector<vector<double>> data;
    int nTrees = 200;
    // cout << data.size();
    // std::ifstream i("./data/soybean.json");
    // json j;
    // i >> j;
    vector<vector<double>> j = readCSV("./data/soybean.data", '\t');
    vector<int> labels;
    int nrows = j.size();
    int nmin = floor(nrows/3);

    cout << nrows << endl;
    int ncols = j[0].size();


    cout << ncols << endl;
    // cout << n_cols + " " + n_rows << "\n";
    // for (auto& element : j) {
    //     vector<double> row;
    //     int counter = 0;
    //     for(auto& element2 : element) {
    //         if (counter < ncols-1) { // The last column is the label one
    //             row.push_back(element2);
    //         }
    //         counter++;
    //     }
    //     data.push_back(row);
    // }
    data = j;
    vector<int> coltypes;
    for (int i=0; i < data[0].size(); i++) {
        coltypes.push_back(1);
    }
    cout << coltypes.size() << endl;
    // vector<vector<double>> matrix;
    MatrixXd matrix = MatrixXd::Zero(nrows, nrows);
    const auto startTime = high_resolution_clock::now();
    #pragma omp parallel for       
    for (int i = 0; i < nTrees; i++) {
        // MatrixXd matrix2(nrows, nrows);

        matrix += build_randomized_tree_and_get_sim(data, nmin, coltypes);
        // matrix += matrix2;

    }
    const auto endTime = high_resolution_clock::now();
    printf("Time: %fms\n", duration_cast<duration<double, milli>>(endTime - startTime).count());
    matrix = matrix/nTrees;
    // int errors = 0;
    
    // std::cout << matrix << std::endl;
    // writeToCSVfile("test.csv", matrix);
    return 1;
}

