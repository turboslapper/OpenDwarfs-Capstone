#CXX      := nvc++
#CXXFLAGS := -acc -gpu=cuda12.3 -O3
CXX := g++
CXXFLAGS := -O3 -std=c++17 -fopenmp -Wall -Wextra
LDFLAGS  :=

# sources and target
SRC      := main.cpp fft_serial.cpp fft_parallel.cpp
OBJ      := $(SRC:.cpp=.o)
TARGET   := fft_app

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) fft_perf.csv

.PHONY: all clean

