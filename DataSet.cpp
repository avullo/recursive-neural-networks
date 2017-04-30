#include "require.h"
#include "General.h"
#include "InstanceParser.h"
#include "DataSet.h"
#include <iostream>
using namespace std;

DataSet::DataSet(const char* fname, bool ownership): own(ownership) {
  ifstream is(fname);
  assure(is, fname);

  int length;
  is >> length;
  require(length, "Dataset size == 0");
  
  InstanceParser parser;
  for(int i=0; i<length; ++i) {
    Instance* instance = parser.read(is);
    require(instance, "Error reading instance");
    push_back(instance);
  }
  
  if(!size()) {
    cerr << "Error! Reading " << fname << " results in an empty data set.";
    require(0, "Aborting...");
  }
}

DataSet::~DataSet() {
  if(own) {
    iterator it = begin();
    while(it != end()) {
      delete *it; *it = 0;
      ++it;
    }
  }
}

void DataSet::add(Instance* i) { push_back(i); }

// shuffle data set instances
void DataSet::shuffle() {
  for (uint k=0; k<size(); ++k) {
    int f1 = rand();
    int f2 = rand();
    int p1 = (int)((double)f1/(1.0+(double)(RAND_MAX))*size());
    int p2 = (int)((double)f2/(1.0+(double)(RAND_MAX))*size());
    
    Instance* tmp = this->operator[](p1);
    (*this)[p1] = (*this)[p2];
    (*this)[p2] = tmp;
  }
}
