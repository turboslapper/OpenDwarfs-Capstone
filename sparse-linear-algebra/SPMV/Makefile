# Makefile for SPMV (three-file build: main.cpp, spmv_serial.cpp, spmv_parallel.cpp)
# Usage:
#   make            # build ./spmv
#   make run-c      # run correctness/timings on hollywood-2009
#   make run-p      # profile all matrices under ../data (writes ../output/spmv_profile.csv)
#   make download   # fetch & extract datasets (expects scripts/get_data.sh)
#   make clean
#
# Note: On very old GCC (e.g., 8.x), you may need:  make LIBS=-lstdc++fs

CXX       ?= g++
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS   ?=
LIBS      ?=

SRCS = main.cpp spmv_serial.cpp spmv_parallel.cpp
OBJS = $(SRCS:.cpp=.o)
BIN  = spmv

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

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

