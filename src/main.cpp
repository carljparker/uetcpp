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
#include <unordered_set>

using json = nlohmann::json;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using namespace std;
using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::milli;

double mean(const std::vector<float> &v)
{
    double sum = 0;

    for (auto &each : v)
        sum += each;

    return sum / v.size();
}

double sd(const std::vector<float> &v)
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
double random_split(vector<float> values, int type)
{

    double split;
    auto result = std::minmax_element(std::begin(values), std::end(values));
    // double sum = std::accumulate(values.begin(), values.end(), 0.0);
    // double mean = sum / values.size();

    // double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    // double stdev = std::sqrt(sq_sum / values.size() - mean * mean);
    srand(time(NULL));
    std::random_device rd;
    std::default_random_engine generator(rd());
    if (type == 0) {
        // cout << mean << " " << stdev << endl;
        // std::normal_distribution<double> distribution(mean, stdev);

        std::uniform_real_distribution<double> distribution(*result.first, *result.second);
        split = distribution(generator);
    }
    else {
        std::uniform_int_distribution<int> dist(0, values.size() - 1);
        split = values[dist(generator)];
    }
        return split;
}

void print(std::vector<float> const &input)
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

vector<float> getColumn(const vector<vector<float>> &v, int attribute) {
    vector<float> col;
    for(auto& row:v) {
        col.push_back(row[attribute]);
        }
    return(col);
    }

vector<vector<float>> getRows(const vector<vector<float>> &v, vector<int> rows) {
    vector<vector<float>>out;
    for(int i:rows) {
            out.push_back(v[i]);
        }
    return(out);
    }

vector<float> remove(std::vector<float> v)
{
	std::vector<float>::iterator itr = v.begin();
	std::unordered_set<int> s;

	for (auto curr = v.begin(); curr != v.end(); ++curr) {
		if (s.insert(*curr).second)
			*itr++ = *curr;
	}

	v.erase(itr, v.end());
    return(v);
}

int performSplit(const vector<vector<float>>& dataset, vector<int>& attributesIndices, 
vector<int> attributes, const vector<int>& coltypes, vector<int> &left, vector<int> &right) {
    int randIndex = rand() % attributesIndices.size();
    int attributeIndex = attributesIndices[randIndex];
    *attributesIndices.erase(attributesIndices.begin()+randIndex); 
    int attribute = attributes[attributeIndex];
    int type = coltypes[attribute];
    vector<float> data;
    vector<vector<float>> localDataset;

    data = getColumn(dataset, attribute);
    
    std::unordered_set<double> set;
    
    for (const double &i: data) {
        set.insert(i);
    }

    if (set.size() == 1){
        return 1;
    }

    double value = random_split(data, type);

    for(int i = 0; i < data.size(); i++) {
        if(data[i] < value) {
            left.push_back(i);
        }
        else {
            right.push_back(i);
        }
    }

    return 0;
}

int performSplit(const vector<vector<float>>& dataset, vector<int>& attributesIndices, 
const vector<int> attributes, const vector<int>& coltypes, vector<int> &left, vector<int> &right, const vector<int> &nodeIndices) {
    int randIndex = rand() % attributesIndices.size();
    int attributeIndex = attributesIndices[randIndex];
    *attributesIndices.erase(attributesIndices.begin()+randIndex); 
    int attribute = attributes[attributeIndex];
    int type = coltypes[attribute];
    vector<float> data;
    vector<vector<float>> localDataset;

    // print(nodeIndices);
    vector<vector<float>> rows = getRows(dataset, nodeIndices);

    data = getColumn(rows, attribute);

    std::unordered_set<double> set;
    for (const double &i: data) {
        set.insert(i);
    }

    if (set.size() == 1){
        return 1;
    }
    double value = random_split(data, type);

    for(int i = 0; i < data.size(); i++) {
        if(data[i] < value) {
                left.push_back(nodeIndices[i]);
        }
        else {
            right.push_back(nodeIndices[i]);
        }
 
    }
    return 0;
}

struct Node {
    vector<int> indices;
    // vector<int> instances;
};

extern "C++"
MatrixXi build_randomized_tree_and_get_sim(const vector<vector<float>>& data, 
const double& nmin, const vector<int>& coltypes) {
    vector<int> seenNodes;
    unordered_set<int> set;
    srand(time(NULL));
    int nrows = data.size();
    int ncols = data.front().size();

    MatrixXi matrix = MatrixXi::Zero(nrows, nrows);
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

    // vector<float> col; // Each column contains rowsize number of elements, i.e, the number of instances 
    vector<int> left_indices, right_indices;
    performSplit(data, attributes_indices, attributes, coltypes, left_indices, right_indices);

    vector<int> left_instances, right_instances;
    vector<vector<float>> left_data, right_data;

    for (int j : left_indices) {
        left_data.push_back(data[j]);
    }   

    for (int j:right_indices) {
        right_data.push_back(data[j]);
    }

    if (left_indices.size() < nmin) {
        for (int instance1:left_indices) {
            seenNodes.push_back(instance1);
            set.insert(instance1);
            for (int instance2:left_indices) {
                matrix(instance1, instance2)+= 1.0;
            } 
        }
    }


    else {
        Node currentNode = {left_indices};
        nodes.push(currentNode);
    }

    if(right_indices.size() < nmin) {
        for (int instance1:right_indices) {
            set.insert(instance1);

            seenNodes.push_back(instance1);
            for (int instance2:right_indices) {
                matrix(instance1, instance2)+= 1.0;   
            }
        }
    }
    else {
        Node currentNode = {right_indices};
        nodes.push(currentNode);
    }

    if (left_indices.size()+right_indices.size() != 150) {
        cout << "Error" << endl;
    }

    // for (int const instance1:left_instances) {
    //     for (int const instance2: right_instances) {
    //         if (instance1 == instance2) {
    //             cout << "Même instance dans deux branches différentes ici !" << endl;
    //         }
    //     }
    // }



    // Root node successfully has two children. Now, we iterate over these children.
    int counter = 0;
    while (!nodes.empty()) {
        // cout << "Loop number " << counter << endl;
        if  (attributes_indices.size() < 1) {
            while (!nodes.empty()) {
                vector<int> instances = nodes.front().indices;
                nodes.pop();
                for (int instance1:instances) {
                    set.insert(instance1);

                    seenNodes.push_back(instance1);
                    for (int instance2:instances) {
                        matrix(instance1, instance2) += 1.0;              
                    }
                }
            }

        break;

        }

        Node currentNode = nodes.front();
        nodes.pop();

        // vector<float> col;
        vector<int> nodeIndices = currentNode.indices;
        if (nodeIndices.size() >= 150) {
            cout << "Error with nodeIndices size" << endl;
        }
        vector<int> left_indices, right_indices;
        // print(left_indices);
        if (nodeIndices.size() >= nmin) {
            if (performSplit(data, attributes_indices, attributes, coltypes, left_indices, right_indices, nodeIndices) == 1) { // We have a column with only one unique value
                for (int instance1:nodeIndices) {
                    set.insert(instance1);

                    seenNodes.push_back(instance1);
                    for (int instance2:nodeIndices) {
                        matrix(instance1, instance2) += 1.0;              
                    }
                }
                cout << "Lower bound number of values reached" << endl;
                continue;                
            }
            // cout << "Nmin not reached" << endl;c
            // print(left_indices);

            vector<int> left_instances, right_instances;
            vector<vector<float>> left_data, right_data;

            for (int j : left_indices) {
                left_data.push_back(data[j]);
            }  
 
            for (int j:right_indices) {
                right_data.push_back(data[j]);
            }

            if (left_instances.size()+right_instances.size()>=150) {
                cout << "Error" << endl;
            }
            if (left_indices.size() < nmin) {
                for (int instance1:left_indices) {
                    seenNodes.push_back(instance1);
                    set.insert(instance1);

                    for (int instance2:left_indices) {
                        matrix(instance1, instance2)+= 1.0;     
                    }
                }
            }

            else {
                Node currentNode = {left_indices};
                nodes.push(currentNode);

            }


            if(right_indices.size() < nmin) {
                for (int instance1:right_indices) {
                    seenNodes.push_back(instance1);
                    set.insert(instance1);

                    for (int instance2:right_indices) {
                        matrix(instance1, instance2)+= 1.0;
                    }
                }

            }

            else {
                Node currentNode = {right_indices};
                nodes.push(currentNode);
            }

        }
        else {
            cout << "Else" << "\n";
        }
        counter++;
    }
    if (seenNodes.size() != 150) {
        cout << seenNodes.size() << endl;
    }
    if (set.size() != 150) {
    cout << set.size() << endl;
    }
    return matrix;
}


vector<vector<float>> readCSV(string filename, char sep) {
    ifstream dataFile;
    dataFile.open(filename);
    vector<vector<float>> csv;
    while(!dataFile.eof()) {
        string line;
        getline(dataFile, line, '\n');
        stringstream buffer(line);
        string tmp;
        vector<float> values;

        while(getline(buffer, tmp, sep) ) {
            values.push_back(strtod(tmp.c_str(), 0));
        }
        csv.push_back(values);
}


    return csv;
}
// #pragma omp declare reduction (+: Eigen::MatrixXi: omp_out=omp_out+omp_in)\
     initializer(omp_priv=MatrixXi::Zero(omp_orig.size(),omp_orig.size()))
int main() {
 
    vector<vector<float>> data;
    int nTrees = 5000;
    // cout << data.size();
    // std::ifstream i("./data/soybean.json");
    // json j;
    // i >> j;
    vector<vector<float>> j = readCSV("./data/iris_wl.csv", ',');
    vector<int> labels;
    int nrows = j.size();
    // print(j[0]);
    // cout << nrows << endl;
    int nmin = floor(nrows/3);
    // nmin = 0;
    // cout << nrows << endl;


    // cout << ncols << endl;
    // cout << n_cols + " " + n_rows << "\n";
    // for (auto& element : j) {
    //     vector<float> row;
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
        coltypes.push_back(0);
    }


    MatrixXi matrix = MatrixXi::Zero(nrows, nrows);
    const auto startTime = high_resolution_clock::now();
    MatrixXi matrices[nTrees];


    
    #pragma omp parallel for
    for (int i = 0; i < nTrees; i++) {
        matrices[i] = build_randomized_tree_and_get_sim(data, nmin, coltypes);
    }
    
    
    for (int i = 0; i < nTrees; i++) {
        matrix += matrices[i];
    }

    const auto endTime = high_resolution_clock::now();
    printf("Time: %fms\n", duration_cast<duration<double, milli>>(endTime - startTime).count());
    MatrixXd matrix2 = matrix.cast <double> ()/nTrees;
    cout << matrix(nrows-1,nrows-1) << endl;
    // for (int i = 0; i < nrows; i++) {
    //     for (int j = 0; j < nrows; j++ ) {
    //         cout << matrix(i,j) << endl;
    //     }
    // }
    // int errors = 0;
    // cout << matrix.size() << endl; 
    ofstream fichier("./matrix.csv", ios::out | ios::trunc);  
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < nrows; j++ ) {
            fichier << matrix2(i,j) << '\t';
        }
        fichier << '\n';
    }
    // fichier << matrix << '\n';
    fichier.close();
    // writeToCSVfile("test.csv", matrix);
    return 1;
}

