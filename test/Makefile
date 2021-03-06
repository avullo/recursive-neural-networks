################################
#                              #
# Unit Tests (Catch framework) #
#                              #
################################

CXXFLAGS += -Wall
CPPFLAGS += -I .. -I 3rdparty/catch
LDFLAGS  = $(wildcard ../*.o)

SOURCES = src/unit.cpp \
	src/unit-dpag.cpp \
	src/unit-domain.cpp \
	src/unit-options.cpp \
	src/unit-instance.cpp \
	src/unit-dataset.cpp

OBJECTS = $(SOURCES:.cpp=.o)

TESTCASES = $(patsubst src/unit-%.cpp,test-%,$(wildcard src/unit-*.cpp))

# main rules

all: $(TESTCASES) rnn_unit

clean:
	rm -rf rnn_unit $(OBJECTS) $(TESTCASES) *~

# single test file

rnn_unit: $(OBJECTS) 3rdparty/catch/catch.hpp
	@echo "[CXXLD] $@"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) -o $@

%.o: %.cpp 3rdparty/catch/catch.hpp
	@echo "[CXX] $@"
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# individual test cases

test-%: src/unit-%.o src/unit.o 3rdparty/catch/catch.hpp
	@echo "[CXXLD] $@"
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $< src/unit.o -o $@

check:
	./rnn_unit

