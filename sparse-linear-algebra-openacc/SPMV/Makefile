## Makefile for SPMV (OpenACC GPU build with NVIDIA HPC SDK)
## Sources: main.cpp, spmv_serial.cpp, spmv_parallel.cpp
## Usage:
##   make            # build ./spmv (OpenACC, GPU)
##   make run-c      # run correctness/timings on hollywood-2009
##   make run-p      # profile all matrices under ../data (writes ../output/spmv_profile.csv)
##   make clean

# OpenACC compiler and flags (example based on your command):
#   nvc++ -acc -gpu=cuda12.3 -O3 vector_add_bench.cpp -o vadd_cpp

# Force use of NVIDIA HPC SDK C++ compiler
CXX      = nvc++

# Core optimization and language flags
BASEFLAGS = -O3 -std=c++17 -Wall

# OpenACC GPU offload flags
ACCFLAGS = -acc -gpu=cuda12.3

# Show accelerator optimization info (optional but useful for tuning)
ACCINFO  = -Minfo=accel

CXXFLAGS ?= $(BASEFLAGS) $(ACCFLAGS) $(ACCINFO)
LDFLAGS  ?= $(ACCFLAGS) $(ACCINFO)
LIBS     ?=

SRCS = main.cpp spmv_serial.cpp spmv_parallel.cpp
OBJS = $(SRCS:.cpp=.o)
BIN  = spmv

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run-c: $(BIN)
	./$(BIN) -c

run-p: $(BIN)
	./$(BIN) -p

run-both: $(BIN)
	./$(BIN) -c -p

download:
	bash scripts/get_data.sh

clean:
	rm -f $(OBJS) $(BIN)

.PHONY: all run-c run-p run-both download clean
