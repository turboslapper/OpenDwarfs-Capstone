# Makefile for SPMV (three-file build: main.cpp, spmv_serial.cpp, spmv_parallel.cpp)
# Usage:
#   make            # build ./spmv
#   make run-c      # run correctness/timings on hollywood-2009
#   make run-p      # profile all matrices under ../data (writes ../output/spmv_profile.csv)
#   make download   # fetch & extract datasets (expects scripts/get_data.sh)
#   make clean
#
# Note: On very old GCC (e.g., 8.x), you may need:  make LIBS=-lstdc++fs

CXX       = g++
CXXFLAGS  = -O3 -std=c++17 -Wall -Wextra -pedantic -fopenmp
LDFLAGS   = -fopenmp

TARGET = spmv
OBJS = main.o spmv_serial.o spmv_parallel.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

spmv_serial.o: spmv_serial.cpp
	$(CXX) $(CXXFLAGS) -c spmv_serial.cpp -o spmv_serial.o

spmv_parallel.o: spmv_parallel.cpp
	$(CXX) $(CXXFLAGS) -c spmv_parallel.cpp -o spmv_parallel.o

run-c: $(TARGET)
	./$(TARGET) -c

run-p: $(TARGET)
	./$(TARGET) -p

run-both: $(TARGET)
	./$(TARGET) -c -p

download:
	bash scripts/get_data.sh

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run-c run-p run-both download clean

