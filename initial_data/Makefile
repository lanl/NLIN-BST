CXX ?= mpicxx
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
TARGET = black_string_id_mpi
SRC = black_string_id_mpi.cpp

all:
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

reader:
	g++ -O2 -std=c++17 -Wall -Wextra -pedantic -o read_bsid_example read_bsid_example.cpp

serial:
	g++ $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) read_bsid_example *.o *.bin *.bin.meta.txt *.csv
