#include "catch.hpp"

#include "Instance.h"
#include "InstanceParser.h"
#include <cstdio>
#include <vector>
#include <fstream>
using namespace std;

typedef unsigned int uint;


TEST_CASE("Basic tests", "[instance]") {
  // prepare arguments and read configuration file
  setenv("RNNOPTIONTYPE", "train", 1);
  char* argv[] = { (char*)"dummy", (char*)"-c", (char*)"data/rnn.conf" };
  Options::instance()->parse_args(3, argv);

  InstanceParser p;
  
  // build and test sequence
  ifstream is("data/sequence.gph");
  Instance* sequence = p.read(is);
  is.close();

  CHECK(sequence->id() == "dummy");
  CHECK(sequence->domain() == SEQUENCE);
  CHECK(sequence->transduction() == IO_ISOMORPH);
  CHECK_FALSE(sequence->output_dim());

  vector<float> node_input, node_target;
  Node* n = sequence->node(0);
  node_input = n->input();
  node_target = n->target();
  CHECK(node_input[0] == .1f); CHECK(node_input[1] == .2f); CHECK(node_input[2] == .3f);
  CHECK(node_target[0] == .3f); CHECK(node_target[1] == .2f); CHECK(node_target[2] == .1f);

  n = sequence->node(4);
  node_input = n->input();
  node_target = n->target();
  CHECK(node_input[0] == .3f); CHECK(node_input[1] == .4f); CHECK(node_input[2] == .5f);
  CHECK(node_target[0] == .5f); CHECK(node_target[1] == .4f); CHECK(node_target[2] == .3f);
  
}
