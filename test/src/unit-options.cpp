#include "catch.hpp"

#include "Options.h"
#include <cstdio>
#include <vector>
#include <iostream>
using namespace std;

typedef unsigned int uint;


TEST_CASE("Basic tests", "[options]") {
  int argc = 10;
  char* argv[] = { (char*)"dummy",
		   (char*)"-c", (char*)"doesnotexist.conf",
		   (char*)"-n", (char*)"test.net",
		   (char*)"-l", (char*)"1e-2",
		   (char*)"-r",
		   (char*)"-o",
		   (char*)"-x" };
  
  // must throw if env variable not set
  CHECK_THROWS(Options::instance()->parse_args(argc, argv));

  // throw if invalid environment variable value
  setenv("RNNOPTIONTYPE", "blablabla", 1);
  CHECK_THROWS(Options::instance()->parse_args(argc, argv));
  
  setenv("RNNOPTIONTYPE", "train", 1);
  
  // throws if cannot find configuration file
  CHECK_THROWS(Options::instance()->parse_args(argc, argv));

  // now point to an existing but incorrect file
  argv[2] = (char*)"data/bad_rnn.conf";
  CHECK_THROWS(Options::instance()->parse_args(argc, argv));
  
  // now point to an existing correct file and check it's read correctly
  argv[2] = (char*)"data/rnn.conf";
  // throws anyway because there's an unknown switch
  CHECK_THROWS(Options::instance()->parse_args(argc, argv));

  argc = 9; // exclude unknown switch from command line parsing
  Options::instance()->parse_args(argc, argv);
  
  CHECK(Options::instance()->usage() == string("dummy [Options]\n"
  					       "Options:\n"
  					       "       -c <global configurations file> (default is ./global.cnf)\n"
  					       "       -n <network file>\n"
  					       "       -l <learning rate> (default is 1e-3)\n"
  					       "       --alpha <momentum coefficient> (default is 1e-1)\n"
  					       "       -ni <regularization coefficient> (default is 0: no regularization)\n"
  					       "       -o on line learning flag (default is batch)\n"
  					       "       -e <number of epochs> (default is 1000)\n"
  					       "       -s <number of epochs between saves of network> (default is 100)\n"
  					       "       -d <data directory>\n"
  					       "       -r training start with random weights (default is read network from file)\n"
  					       "       --train-set <training set file in data dir>\n"
  					       "       --test-set  <test set file in data directory>\n"
  					       "       --validation-set <validation set file in data directory>\n"
  					       "       --threshold-error <threshold error to be used to stop training> (default is 1e-3)\n"));
  // check values read from configuration file
  CHECK(Options::instance()->domain() == SEQUENCE);
  CHECK(Options::instance()->transduction() == IO_ISOMORPH);
  CHECK(Options::instance()->input_dim() == 3);
  CHECK(Options::instance()->output_dim() == 3);
  CHECK(Options::instance()->domain_outdegree() == 5);
  pair<int, int> li = Options::instance()->layers_indices();
  CHECK(li.first == 2); CHECK(li.second == 1);
  vector<int> lnu = Options::instance()->layers_number_units();
  CHECK(lnu[0] == 10); CHECK(lnu[1] == 5); CHECK(lnu[2] == 5);

  // check application specific configuration values
  CHECK(atof(Options::instance()->get_parameter("eta").c_str()) == 1e-2);
  CHECK(atof(Options::instance()->get_parameter("alpha").c_str()) == .1);
  CHECK(atof(Options::instance()->get_parameter("ni").c_str()) == .0);
  CHECK(atoi(Options::instance()->get_parameter("epochs").c_str()) == 1000);
  CHECK(atoi(Options::instance()->get_parameter("savedelta").c_str()) == 100);
  CHECK(atoi(Options::instance()->get_parameter("onlinelearning").c_str()) == 1);
  CHECK(atoi(Options::instance()->get_parameter("random_net").c_str()) == 1);
  CHECK(atof(Options::instance()->get_parameter("threshold_error").c_str()) == 1e-3);
}
