#############################
#                           #
# Makefile for GNU compiler #
#                           #
#############################

DEFINES  = -DDEBUG
# Dependency on Boost library
INCLUDES = -I/usr/include/boost
WARNINGS = -Wall
CODEOPT  = # -ftemplate-depth-30 -fpermissive -O3
DEBUG    = -g
CXXFLAGS = $(DEFINES) $(WARNINGS)
CPPFLAGS = $(INCLUDES) $(CODEOPT) $(DEBUG)
PROFILE  = # -pg
LDFLAGS  =
LIBS     = 
LD       = $(CXX)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(PROFILE) -c $< -o $@

SOURCES.cpp = \
	DPAG.cpp \
	DataSet.cpp \
	Instance.cpp \
	InstanceParser.cpp \
	Node.cpp \
	Options.cpp \
	StructuredDomain.cpp

SOURCES.h= \
	ActivationFunction.h \
	DPAG.h \
	DataSet.h \
	ErrorMinimizationProcedure.h \
	General.h \
	Instance.h \
	InstanceParser.h \
	Node.h \
	Options.h \
	RecurisveNN.h \
	StructuredDomain.h \
	require.h

OBJECTS = $(SOURCES.cpp:%.cpp=%.o)

TARGETS = rnnTrain generateParityGraphs

# main targets
all: ${TARGETS}

rnnTrain:  $(OBJECTS) rnnTrain.o
	$(LD) $(OBJECTS) rnnTrain.o $(LIBS) -o $@ $(PROFILE) $(LDFLAGS)

rnnTrain.o:  $(SOURCES.cpp) rnnTrain.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(PROFILE) -c rnnTrain.cpp -o $@

generateParityGraphs: generateParityGraphs.o
	$(LD) generateParityGraphs.o -o $@ $(PROFILE) $(LDFLAGS)

clean:
	rm -rf *.o *.bak *~ $(TARGETS)
	$(MAKE) clean -C test

# unit tests
## build unit tests into one single command
tests: ${OBJECTS}
	$(MAKE) rnn_unit -C test

## run unit tests
check:
	$(MAKE) check -C test

depend:
	makedepend -- $(CXXFLAGS) $(CPPFLAGS) rnnTrain.cpp generateParityGraphs.cpp --
