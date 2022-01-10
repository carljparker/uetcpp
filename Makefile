GPP=/usr/local/Cellar/gcc/11.2.0_3/bin/g++-11

 
main: src/main.cpp
	$(GPP) -fopenmp -std=c++17 src/main.cpp -o main -I./include -I./Eigen


# --- END --- #
 
