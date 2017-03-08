#include "catch.hpp"

#include "Options.h"
#include <cstdio>
#include <vector>
#include <iostream>
using namespace std;

typedef unsigned int uint;


TEST_CASE("Basic tests", "[options]") {
  // test default, just read from configuration file
  int argc = 3;
  char* argv[] = { (char*)"dummy", (char*)"-c", (char*)"doesnotexist.conf" };
  setenv("RNNOPTIONTYPE", "blablabla", 1);

  // throws if cannot find configuration file
  CHECK_THROWS(Options::instance()->parse_args(argc, argv));

  // now point to an existing file
  argv[2] = (char*)"data/rnn.conf";
  Options::instance()->parse_args(argc, argv);
  CHECK(Options::instance()->usage() == string("dummy [Options]\nOptions:\n\t-c <config file> (default is .rnnrc)\n"));
   
}
