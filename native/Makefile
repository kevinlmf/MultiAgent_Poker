# OpenMP parallel greedy capacity solver
CXX ?= c++
CXXFLAGS = -O3 -std=c++17 -fPIC -shared

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LIB = libcapacity_greedy.dylib
  LIBOMP := $(shell brew --prefix libomp 2>/dev/null)
  ifneq ($(LIBOMP),)
    OMPFLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP)/include -L$(LIBOMP)/lib -lomp
  else
    OMPFLAGS =
  endif
else
  LIB = libcapacity_greedy.so
  OMPFLAGS = -fopenmp
endif

all: $(LIB)

$(LIB): capacity_greedy.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) capacity_greedy.cpp -o $(LIB)

clean:
	rm -f libcapacity_greedy.so libcapacity_greedy.dylib

.PHONY: all clean
