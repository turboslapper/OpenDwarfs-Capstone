CXX      := g++
CXXFLAGS := -g -O3 -std=c++17
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

