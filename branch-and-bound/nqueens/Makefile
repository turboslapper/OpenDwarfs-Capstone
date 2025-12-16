CXX      := g++
CXXFLAGS := -O3 -std=c++17 -fopenmp -Wall -Wextra
#CXX      := nvc++
#CXXFLAGS := -acc -gpu=cuda12.3 -O3
TARGET   := nq
SRCS     := main.cpp nq_serial.cpp nq_parallel.cpp
OBJS     := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run-correctness: $(TARGET)
	./$(TARGET) -c

run-perf: $(TARGET)
	./$(TARGET) -p > results.csv
	@echo "CSV written to results.csv"

clean:
	rm -f $(OBJS) $(TARGET) results.csv