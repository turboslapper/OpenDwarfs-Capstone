# Makefile
CXX       ?= g++
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra -pedantic
# If you're on GCC 8, you may need: make LIBS=-lstdc++fs
LIBS      ?=

SRCS = main.cpp bfs_serial.cpp bfs_parallel.cpp
OBJS = $(SRCS:.cpp=.o)
BIN  = bfs

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

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

