#############################
#                           #
# Makefile for GNU compiler #
#                           #
#############################

DEFINES  = -DDEBUG
# Dependency on Boost library
INCLUDES = -I/usr/include/boost
WARNINGS = -Wall
CODEOPT  = # -O3
DEBUG    = -g
CXXFLAGS = $(WARNINGS)
CPPFLAGS = $(DEFINES) $(INCLUDES) $(CODEOPT) $(DEBUG)
LDFLAGS  =

.cpp.o:
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

SOURCES = DPAG.cpp StructuredDomain.cpp Options.cpp

OBJECTS = $(SOURCES:.cpp=.o)

# main targets
# NOTE: just compile into objects until we have some mains
all: $(OBJECTS)

clean:
	rm -rf *.o *~
	$(MAKE) clean -C test

# unit tests
## build unit tests into one single command
tests:
	$(MAKE) rnn_unit -C test

## run unit tests
check:
	$(MAKE) check -C test
