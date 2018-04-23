#include "require.h"
#include "General.h"
#include "Options.h"
#include "Performance.h"

#include <cassert>
#include <algorithm>
using namespace std;

Performance::Performance(): _problem(Options::instance()->problem()) {
  // get types of trasductions to implement
  Transduction tr = Options::instance()->transduction();
  switch(tr) {
  case IO_ISOMORPH:
    _ios_tr = true;
    _ss_tr  = false;
    break;
  case SUPER_SOURCE:
    _ios_tr = false;
    _ss_tr  = true;
    break;
  default:
    require(0, "Unknown transduction type");
  }

}

Performance* Performance::factory(const Problem& problem)
  throw(BadPerformanceCreation) {
  if(problem & (BINARYCLASS | MULTICLASS)) {
    return new ClassificationPerformance(Options::instance()->output_dim()==1?2:Options::instance()->output_dim());
  }
  if(problem & REGRESSION)
    return new RegressionPerformance();

  throw BadPerformanceCreation("Unknown problem type");

}

ClassificationPerformance::ClassificationPerformance(int nclasses): Performance(), _nc(nclasses) {
  _cm = new int*[_nc];
  for(int c=0; c<_nc; ++c)
    _cm[c] = new int[_nc];

  reset();
}

ClassificationPerformance::~ClassificationPerformance() {
  for(int c=0; c<_nc; ++c)
    delete[] _cm[c];
  delete[] _cm;
}

void ClassificationPerformance::reset() {
  for(int c1=0; c1<_nc; ++c1)
    for(int c2=0; c2<_nc; ++c2)
      _cm[c1][c2] = 0;
}

void ClassificationPerformance::update(Instance* instance) {
  if(_ss_tr) {
    // super-source transduction
    // compare instance global predicted output with target
    vector<float> target = instance->target();
    vector<float> output = instance->output();

    // problem can be multiple indepedent binary classification.
    // In such cases there are multiple outputs
    //
    // NOTE: split into multiple confusion matrices?
    //
    if(_problem & BINARYCLASS)
      for(uint i=0; i<output.size(); ++i)
	_cm[output[i]>=.5?1:0][target[i]>=.5?1:0]++;
    else
      // multiclass problem: softmax
      _cm[distance(output.begin(), max_element(output.begin(), output.end()))][distance(target.begin(), max_element(target.begin(), target.end()))]++;
    
    return;
  }

  // io-isomorph transduction
  // compare each node predicted output with its target
  if(_ios_tr)
    for(uint n=0; n<instance->num_nodes(); ++n) {
      Node* node = instance->node(n);
    
      vector<float> target = node->target();
      vector<float> output = node->output();
      
      if(_problem & BINARYCLASS)
	for(uint i=0; i<output.size(); ++i)
	  _cm[output[i]>=.5?1:0][target[i]>=.5?1:0]++;
      else
	// multiclass problem: softmax
	_cm[distance(output.begin(), max_element(output.begin(), output.end()))][distance(target.begin(), max_element(target.begin(), target.end()))]++;
      
    }
}

void ClassificationPerformance::print(ostream& os) {
  int* nclassinst = new int[_nc];
  memset(nclassinst, 0, _nc * sizeof(int));
  
  int tot_insts = 0;
  for(int c=0; c<_nc; ++c) {
    nclassinst[c] = num_class_instances(c);
    tot_insts += nclassinst[c];
  }

  int tot_class_errors = num_total_errors();
  os << "NErrors " << tot_class_errors << "/" << tot_insts;
  os << '\t' << (1 - (double)tot_class_errors/(double)(tot_insts)) * 100.0 << '%'
     << endl;
  
  for(int c=0; c<_nc; ++c) {
    int tot_errors_per_class = num_class_errors(c);
    os << "Class" << c << ' ' << tot_errors_per_class << "/" << nclassinst[c];
    os << "\t" << (1 - (double)tot_errors_per_class/(double)nclassinst[c]) * 100.0 << '%'
       << endl;
  }

  os << endl;
}

int ClassificationPerformance::num_class_instances(int c) const {
  assert(c >= 0 && c < _nc);

  int count = 0;
  for(int i=0; i<_nc; ++i)
    count += _cm[i][c];

  return count;
}

int ClassificationPerformance::num_class_errors(int c) const {
  assert(c >= 0 && c < _nc);
  
  int count = 0;
  for(int i=0; i<_nc; ++i)
    if(i != c)
      count += _cm[i][c];
  
  return count;
}

int ClassificationPerformance::num_total_errors() const {
  int count = 0;
  for(int c1=0; c1<_nc; ++c1)
    for(int c2=0; c2!=_nc; ++c2) 
      if(c1 != c2)
	count += _cm[c1][c2];
  
  return count;
  
}
