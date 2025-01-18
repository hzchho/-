#!/bin/bash

# 设置源文件和目标文件的名称
SRC_MAIN="before_optimize.cpp"
SRC_MAIN2="after_optimize.cpp"
EXEC_FILE="before_optimize"
EXEC_FILE2="after_optimize"

# compile and execute
echo "g++ -O0 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN -o $EXEC_FILE && ./$EXEC_FILE 100"
g++ -O0 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN -o $EXEC_FILE && ./$EXEC_FILE 100

echo "g++ -O2 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN -o $EXEC_FILE && ./$EXEC_FILE 100"
g++ -O2 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN -o $EXEC_FILE && ./$EXEC_FILE 100

echo "g++ -O3 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN -o $EXEC_FILE && ./$EXEC_FILE 100"
g++ -O3 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN -o $EXEC_FILE && ./$EXEC_FILE 100

echo "g++ -O0 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN2 -o $EXEC_FILE2 && ./$EXEC_FILE2 100"
g++ -O0 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN2 -o $EXEC_FILE2 && ./$EXEC_FILE2 100

echo "g++ -O2 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN2 -o $EXEC_FILE2 && ./$EXEC_FILE2 100"
g++ -O2 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN2 -o $EXEC_FILE2 && ./$EXEC_FILE2 100

echo "g++ -O3 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN2 -o $EXEC_FILE2 && ./$EXEC_FILE2 100"
g++ -O3 -march=native -mavx2 -std=c++23 -fopenmp $SRC_MAIN2 -o $EXEC_FILE2 && ./$EXEC_FILE2 100