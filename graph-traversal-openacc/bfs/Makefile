# Makefile for BFS (three-file build: main.cpp, bfs_serial.cpp, bfs_parallel.cpp)
# OpenACC (NVHPC) version for glogin.

CXX = nvc++

ACCFLAGS = -acc -gpu=cuda12.3 -Minfo=accel
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -pedantic $(ACCFLAGS)
LDFLAGS  = $(ACCFLAGS)
LIBS     =

SRCS = main.cpp bfs_serial.cpp bfs_parallel.cpp
OBJS = $(SRCS:.cpp=.o)
BIN  = bfs

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
