
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
#include <Eigen/Eigen>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <unordered_set>
#include <cxxopts.hpp>

using Eigen::MatrixXd;

using namespace std;
using std::milli;
using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::sort;


float random_split(vector<float> values, int type)
{
  float split;
  auto result = std::minmax_element(std::begin(values), std::end(values));
  
  srand(time(NULL));
  std::random_device rd;
  std::mt19937_64 generator(rd());
  if (type == 0)
  {
    std::uniform_real_distribution<float> distribution(*result.first, *result.second);
    split = distribution(generator);
  }
  else
  { 
    std::uniform_int_distribution<int> distribution(0, values.size()-1);  
    split = values[distribution(generator)];
  }
  
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
                 const vector<int> &attributes, const vector<int> &coltypes, vector<int> &left, vector<int> &right, const vector<int> &nodeIndices)
{
  
  std::random_device rd;
  std::mt19937_64 mt(rd());
  std::uniform_int_distribution<int> dist (0,attributesIndices.size()-1);
  
  //Sample a random index column
  int randIndex = dist(mt);
  
  //Get the corresponding index attribute
  int attributeIndex = attributesIndices[randIndex];
  
  *attributesIndices.erase(attributesIndices.begin() + randIndex);
  int attribute = attributes[attributeIndex];
  //Get the corresponding column type
  int type = coltypes.at(attribute);
  vector<vector<float>> localDataset = getRows(dataset, nodeIndices);
  vector<float> data;

  data = getColumn(localDataset, attribute);
  
  std::unordered_set<float> set;
  
  //Insert all the column values in a set
  for (const float &i : data)
  {
   set.insert(i);
  }
  
  //If there is only one value:
  if (set.size() <= 1)
  {
    return 1;
  }
  //Split the data according to its type
  float value = random_split(data, type);
  
  
  //Fill the indices

  // if (type == 0) { // Continuous attribute
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
  // }
  // else {
  //     for (int i = 0; i < data.size(); i++) {
  //       if (data[i] == value)
  //       {
  //         left.push_back(i);
  //       }
  //       else
  //       {
  //         right.push_back(i);
  //       }
  //   }  
  // }

  return 0;
}

struct Node
{
  vector<int> indices;
  vector<int> instances;
};

MatrixXd getSim(const vector<vector<float>> &data,
                                           const int &nmin, const vector<int> &coltypes, int nTrees)
{
  int nrows = data.size();
  vector<float> firstVector=data[0];
  int ncols=firstVector.size();
  MatrixXd matrix = MatrixXd::Zero(nrows, nrows);
  vector<int> nodeIndices;

  for (int i=0; i < nrows; i++) {
    nodeIndices.push_back(i);
  }
  #pragma omp parallel for
  for (int loop = 0; loop < nTrees; loop++)
  {
    list<Node> nodes;
    vector<int> attributes, attributes_indices, instanceList;
    for (int i = 0; i < ncols; i++)
    {
      attributes_indices.push_back(i);
      attributes.push_back(i);
    }

    for (int i=0; i < nrows; i++) {
      instanceList.push_back(i);
    }

    vector<int> left_indices, right_indices, left_instances, right_instances;
    performSplit(data, attributes_indices, attributes, coltypes, left_indices, right_indices, nodeIndices);
    for (int i : left_indices) {
      left_instances.push_back(instanceList[i]);
    }
    for (int i : right_indices) {
      right_instances.push_back(instanceList[i]);
    }
    if (left_indices.size() < nmin)
    {
      for (int instance1 : left_instances)
      {
        for (int instance2 : left_instances)
        {
          #pragma omp atomic
          matrix(instance1, instance2) += 1.0/nTrees;
        }
      }
    }

    else
    {
      Node currentNode = {left_indices, left_instances};
      nodes.push_back(currentNode);
    }

    if (right_indices.size() < nmin)
    {
      for (int instance1 : right_instances)
      {

        for (int instance2 : right_instances)
        {
          #pragma omp atomic
          matrix(instance1, instance2) += 1.0/nTrees;
        }
      }
    }
    else
    {
      Node currentNode = {right_indices, right_instances};
      nodes.push_back(currentNode);
    }

    // Root node successfully has two children. Now, we iterate over these children.
    while (!nodes.empty())
    {
      if (attributes_indices.size() < 1)
      {
        for (Node node : nodes)
        {
          vector<int> instances = node.instances;
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
      nodes.pop_front();

      vector<int> nodeIndices = currentNode.indices;
      vector<int> nodeInstances = currentNode.instances;
      vector<int> left_indices, right_indices, left_instances, right_instances;
      int colNum = performSplit(data, attributes_indices, attributes, coltypes, left_indices, right_indices, nodeIndices);

      if (colNum == 1) // We have a column with only one unique value
      {
        for (int instance1 : nodeInstances)
        {
          for (int instance2 : nodeInstances)
          {
            #pragma omp atomic
            matrix(instance1, instance2) += 1.0/nTrees;
          }
        }
      }
      else 
      {
        for (int i : left_indices) {
          left_instances.push_back(nodeInstances[i]);
        }
        for (int i : right_indices) {
          right_instances.push_back(nodeInstances[i]);
        }

        if (left_indices.size() < nmin)
        {
          for (int instance1 : left_instances)
          {
            for (int instance2 : left_instances)
            {
              #pragma omp atomic
              matrix(instance1, instance2) += 1.0/nTrees;
            }
          }
        }

        else
        {
          Node currentNode = {left_indices, left_instances};
          nodes.push_back(currentNode);
        }

        if (right_indices.size() < nmin)
        {
          for (int instance1 : right_instances)
          {
            for (int instance2 : right_instances)
            {
              #pragma omp atomic
              matrix(instance1, instance2) += 1.0/nTrees;
            }
          }
        }

        else
        {
          Node currentNode = {right_indices, right_instances};
          nodes.push_back(currentNode);
        }
      }
    }
  }
  return matrix;
}

MatrixXd getDist(const vector<vector<float>> &data,
                                           const int &nmin, const vector<int> &coltypes, int nTrees)
{
  int nrows = data.size();
  vector<float> firstVector=data[0];
  int ncols=firstVector.size();
  MatrixXd matrix = MatrixXd::Zero(nrows, nrows);
  vector<int> nodeIndices;

  for (int i=0; i < nrows; i++) {
    nodeIndices.push_back(i);
  }
  #pragma omp parallel for
  for (int loop = 0; loop < nTrees; loop++)
  {
    list<Node> nodes;
    vector<int> attributes, attributes_indices, instanceList;
    for (int i = 0; i < ncols; i++)
    {
      attributes_indices.push_back(i);
      attributes.push_back(i);
    }

    for (int i=0; i < nrows; i++) {
      instanceList.push_back(i);
    }

    vector<int> left_indices, right_indices, left_instances, right_instances;
    performSplit(data, attributes_indices, attributes, coltypes, left_indices, right_indices, nodeIndices);
    for (int i : left_indices) {
      left_instances.push_back(instanceList[i]);
    }
    for (int i : right_indices) {
      right_instances.push_back(instanceList[i]);
    }
    if (left_indices.size() < nmin and left_instances.size() != 0)
    {
      for (int instance1 : left_instances)
      {
        for (int instance2 : left_instances)
        {
          matrix(instance1, instance2) += left_instances.size();
        }
      }
    }

    else
    {
      Node currentNode = {left_indices, left_instances};
      nodes.push_back(currentNode);
    }

    if (right_indices.size() < nmin and right_instances.size() != 0)
    {
      for (int instance1 : right_instances)
      {

        for (int instance2 : right_instances)
        {
          matrix(instance1, instance2) += right_instances.size();
        }
      }
    }
    {
      Node currentNode = {right_indices, right_instances};
      nodes.push_back(currentNode);
    }
    
    for (int instance1 : left_instances) {
        for (int instance2 : right_instances) {
              matrix(instance2,instance1) = matrix(instance1,instance2) += (left_instances.size() + right_instances.size());

        }
    }
    // Root node successfully has two children. Now, we iterate over these children.
    while (!nodes.empty())
    {
      if (attributes_indices.size() < 1)
      {
        for (Node node : nodes)
        {
          vector<int> instances = node.instances;
          for (int instance1 : instances)
          {
            for (int instance2 : instances)
            {
              matrix(instance1, instance2) += 1;
            }
          }
        }
        break;
      }

      Node currentNode = nodes.front();
      nodes.pop_front();

      vector<int> nodeIndices = currentNode.indices;
      vector<int> nodeInstances = currentNode.instances;
      vector<int> left_indices, right_indices, left_instances, right_instances;
      int colNum = performSplit(data, attributes_indices, attributes, coltypes, left_indices, right_indices, nodeIndices);

      if (colNum == 0)
      {
      for (int i : left_indices) {
        left_instances.push_back(nodeInstances[i]);
      }
      for (int i : right_indices) {
        right_instances.push_back(nodeInstances[i]);
      }

      if (left_indices.size() < nmin and left_indices.size() != 0)
      {
        for (int instance1 : left_instances)
        {
          for (int instance2 : left_instances)
          {
            matrix(instance1, instance2) += left_instances.size();
          }
        }
      }

      else
      {
        Node currentNode = {left_indices, left_instances};
        nodes.push_back(currentNode);
      }

      if (right_indices.size() < nmin and right_indices.size() != 0)
      {
        for (int instance1 : right_instances)
        {
          for (int instance2 : right_instances)
          {
            matrix(instance1, instance2) += right_instances.size();
          }
        }
      }

      else
      {
        Node currentNode = {right_indices, right_instances};
        nodes.push_back(currentNode);
      }
    }

        for (int instance1 : left_instances) {
            for (int instance2 : right_instances) {
              matrix(instance2,instance1) = matrix(instance1,instance2) += (left_instances.size() + right_instances.size());

        }
    }
  }
  }
  return matrix/(nrows*nTrees);
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
int getNmin (vector<double> stds) {
  int minIndex  = std::min_element(stds.begin(), stds.end()) - stds.begin();
  return minIndex;
}

int main(int argc, char** argv)
{

    cxxopts::Options options("uetcpp", "An implementation of UET");
    options.add_options()
    ("p,path", "Data path", cxxopts::value<string>())
    ("s,sep", "Separator", cxxopts::value<char>()->default_value("\t"))
    ("c,ctypes", "Coltypes",  cxxopts::value<string>())
    ("n,nmin", "Nmin", cxxopts::value<float>()->default_value("0.33"))
    ("t,ntrees", "Number of trees",  cxxopts::value<int>()->default_value("500"))
    ("m,massbased", "Mass-based dissimilarity",  cxxopts::value<int>()->default_value("0"))
    ("o,optimize", "Find optimal parameters", cxxopts::value<int>()->default_value("0"));
    auto result = options.parse(argc, argv);
    string path = result["path"].as<string>();
    char sep = result["sep"].as<char>();
    
    string coltypesString = result["ctypes"].as<string>();
    float nminPercent = result["nmin"].as<float>();
    int nTrees = result["ntrees"].as<int>();
    int massBased = result["massbased"].as<int>();
    vector<vector<float>> data;
    data = readCSV(path, sep);
    int nrows = data.size();
    int ncols = data[0].size();
    int nmin = floor(nminPercent*nrows);
    vector<int> coltypes;
    if (coltypesString.back() == ',') { // All attributes are of the same type
      if (coltypesString.front() == '0') {
        for (int i=0; i <= ncols; i++) {
          coltypes.push_back(0);
        }
      }
      else {
        for (int i=0; i <= ncols; i++) {
          coltypes.push_back(1);
        }
      } 
    }
    else {
      for (int i : coltypesString) {
        coltypes.push_back(i-48);
      }
    }

    data.pop_back();

    MatrixXd matrix;
    if(result["optimize"].as<int>() == 1) {
      vector<double> std_devs;
      vector<int> nmins = {0.05*nrows, 0.10*nrows, 0.15*nrows, 0.20*nrows, 0.25*nrows, 0.3*nrows, 0.35*nrows, 0.40*nrows,0.45*nrows, 0.50*nrows, 0.55*nrows, 0.60*nrows};
      vector<int> ntrees = {50, 100, 200, 500, 1000, 2000, 10000};
      for (int nmin :  nmins) {
        if (massBased == 0) {
            matrix = getSim(data, nmin, coltypes, 500);
            MatrixXd mean = MatrixXd::Constant(data.size(), data.size(), matrix.mean());
            float sd = sqrt(1/(matrix.array() - mean.array()).pow(2).sum());
            std_devs.push_back(sd);
        }
        else {
            matrix = getDist(data, 0, coltypes, nTrees);
        }
      }
      nmin = nmins.at(getNmin(std_devs));
      cout << nmin << endl;
      float previousSd = 0;
      for (int ntree : ntrees) {
        if (massBased == 0) {
            matrix = getSim(data, nmin, coltypes, ntree);
            MatrixXd mean = MatrixXd::Constant(data.size(), data.size(), matrix.mean());
            float sd = sqrt(1/(matrix.array() - mean.array()).pow(2).sum());
            if (abs(sd-previousSd) < 0.005) {
              nTrees = ntree;
              cout << ntree << endl;
              break;
            }

            previousSd = sd;
            std_devs.push_back(sd);
        }
        else {
            matrix = getDist(data, 0, coltypes, nTrees);
        }      
      }
    }


    const auto startTime = high_resolution_clock::now();
      if (massBased == 0) {
          matrix = getSim(data, nmin, coltypes, nTrees);
      }
      else {
          matrix = getDist(data, 0, coltypes, nTrees);
      }



    const auto endTime = high_resolution_clock::now();

    printf("Time: %fms\n", duration_cast<duration<double, milli>>(endTime - startTime).count());


    ofstream fichier("./matrix_uet.csv", ios::out | ios::trunc);
    for (int i = 0; i < nrows-1; i++)
    {
        for (int j = 0; j < nrows-1; j++)
        {
            fichier << matrix(i, j) << '\t';
        }
        fichier << '\n';
    }
    fichier.close();
    // writeToCSVfile("test.csv", matrix);

    return 1;
}
