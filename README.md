# UETcpp

To compile, use the following command: 

g++ -fopenmp -std=c++17 main.cpp -o main -I./include -I./Eigen


# macOS Build instructions #

Tested on: macOS 11.6 (20G165)

Install a proper GNU C++ Compiler using Homebrew

    brew install gcc

In modern versions of macOS `g++` 1) calls clang and 2) doesn't support
the `-fopenmp` option.



# References #

> Dalleau, Kevin, Miguel Couceiro, and Malika Smail-Tabbone. 
> “Unsupervised Extra Trees: A Stochastic Approach to Compute Similarities in Heterogeneous Data.” 
> International Journal of Data Science and Analytics 9, no. 4 (May 1, 2020): 447–59.
> https://doi.org/10.1007/s41060-020-00214-4.


### --- END --- ###

