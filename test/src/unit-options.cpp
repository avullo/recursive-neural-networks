/*
 * Recursive Neural Networks: neural networks for data structures 
 *
 * Copyright (C) 2018 Alessandro Vullo 
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/

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
  					       "       -c <global configurations file> (default: .rnnrc in current directory)\n"
  					       "       -n <network file> [REQUIRED]\n"
  					       "       -l <learning rate> (default is 1e-3)\n"
  					       "       --alpha <momentum coefficient> (default is 1e-1)\n"
  					       "       -ni <regularization coefficient> (default is 0: no regularization)\n"
  					       "       -o on line learning flag (default is batch)\n"
  					       "       -e <number of epochs> (default is 1000)\n"
  					       "       -s <number of epochs between saves of network> (default is 100)\n"
  					       "       -r training start with random weights (default is read network from file)\n"
  					       "       --training-set <training set file> [REQUIRED]\n"
  					       "       --test-set  <test set file> [OPTIONAL]\n"
  					       "       --validation-set <validation set file> [OPTIONAL]\n"
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
  CHECK(lnu.size() == li.first + li.second);
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
