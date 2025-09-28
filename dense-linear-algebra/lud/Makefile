# === Makefile for LUD (dense, 7-point Laplacian) ===

# Compiler and flags
CXX      ?= g++
STD      ?= -std=c++17
OPT      ?= -O3 -fopenmp
WARN     ?= -Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion
DEBUGSYM ?= -g
SANFLAGS ?=
# Enable sanitizers with: make SAN=1
ifeq ($(SAN),1)
  SANFLAGS += -fsanitize=address,undefined -fno-omit-frame-pointer
endif

CXXFLAGS ?= $(STD) $(OPT) $(WARN) $(DEBUGSYM) $(SANFLAGS)
LDFLAGS  ?= $(SANFLAGS)

# Files and binary
SRC := main.cpp lud_serial.cpp lud_parallel.cpp
OBJ := $(SRC:.cpp=.o)
BIN := lud

# Default target
.PHONY: all
all: $(BIN)

# Build rules
$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Config targets
.PHONY: debug release
debug: OPT := -O0
debug: $(BIN)

release: OPT := -O3 -march=native
release: $(BIN)

# Convenience runs
.PHONY: run compare perf
run: $(BIN)
	./$(BIN) -c

compare: $(BIN)
	./$(BIN) -c

perf: $(BIN)
	./$(BIN) -p
	@echo "CSV should be in lud_perf.csv"

# Clean
.PHONY: clean
clean:
	$(RM) -f $(OBJ) $(BIN) lud_perf.csv

# Print variables (for debugging Make)
.PHONY: vars
vars:
	@echo "CXXFLAGS = $(CXXFLAGS)"
	@echo "LDFLAGS  = $(LDFLAGS)"
	@echo "SRC      = $(SRC)"
	@echo "OBJ      = $(OBJ)"
	@echo "BIN      = $(BIN)"

