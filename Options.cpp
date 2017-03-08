#include "require.h"
#include "Options.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iostream>
#include <string>
using namespace std;

void Options::parse_args(int argc, char* argv[]) 
  throw(Options::BadOptionSetting) {

  // prepend program name to usage string
  _usage = string(argv[0]) + " " + _usage;
  
  // parse command line switches, search only for global configuration file option
  // derived classes will consider application specific flags
  for(int i=1; i<argc; i++) {
    string option(argv[i]);
    
    if(option == "-c") {
      args["config"] = string(argv[i+1]);
      continue;
    }
  }
  // echo usage in case required application parameters are not given
  map<string, string>::const_iterator it;
  it = args.find("config");
  if(it == args.end())
    throw BadOptionSetting(string("Must specify global configuration file; use -c <config file> command line flag"));

  // read global configuration file and set globally visible parameters
  ifstream ifs((*it).second.c_str());
  assure(ifs, (*it).second.c_str());
  
  string line, dummy;
  while(getline(ifs, line)) {
    if(line[0] == '#') continue; // skip comments

    int pos;
    istringstream iss(line);
    
    pos = line.find("input_dimension");
    if(pos != string::npos) {
      iss >> dummy >> _input_dimension;
      assert(_input_dimension > 0);
      continue;
    }
    
    pos = line.find("domain_outdegree");
    if(pos != string::npos) {
      iss >> dummy >> _domain_outdegree;
      assert(_domain_outdegree > 0);
      continue;
    }
    
    pos = line.find("layers_number_units");
    if(pos != string::npos) {
      iss >> dummy >> _r >> _s;
      int i = 0, lnu;
      while(i<_r+_s) {
	iss >> lnu;
	_lnunits.push_back(lnu);
	++i;
      }
      continue;
    }

    pos = line.find("rnn_weights_precision");
    if(pos != string::npos) {
      iss >> dummy >> _precision;
      assert(_precision > 0);
      continue;
    }

    pos = line.find("domain");
    if(pos != string::npos) {
      string d;
      iss >> dummy >> d;
      if(d == "DOAG") { _domain = DOAG; }
      else if(d == "SEQUENCE") { _domain = SEQUENCE; }
      else if(d == "LINEARCHAIN") { _domain = LINEARCHAIN; }
      else if(d == "NARYTREE") { _domain = NARYTREE; }
      else if(d == "UG") { _domain = UG; }
      else if(d == "GRID2D") { _domain = GRID2D; }
      else { require(0, "Unrecognised domain type"); }
      continue;
    }

    pos = line.find("transduction");
    if(pos != string::npos) {
      string trans;
      iss >> dummy >> trans;

      if(trans == "SUPER_SOURCE") { _transduction = SUPER_SOURCE; }
      else if(trans == "IO ISOMORPH") { _transduction = IO_ISOMORPH; }
      else { require(0, "Transduction type not recognized"); }
      continue;
    }
  }

  // TODO: raise exception if config parameters are not set,
  //       or add default values
}

void RNNTrainingOptions::parse_args(int argc, char* argv[]) 
  throw(Options::BadOptionSetting) {

  // parse connfiguration file first
  try {
    Options::parse_args(argc, argv);
  } catch(Options::BadOptionSetting e) {
    cerr << e.explainError() << endl << endl;
    cerr << usage << endl;
    exit(EXIT_FAILURE);
  }
 
  /*** parse command line and find application specific options ***/
  for (int i = 1; i < argc; i++) {
    // parse switches
    if (argv[i][0] == '-') {
      string arg(argv[i]);
      if(arg == "-n") {
	args["netname"] = string(argv[++i]);
      } else if(arg == "-l") {
	args["eta"] = string(argv[++i]);
      } else if(arg == "--alpha") {
	args["alpha"] = string(argv[++i]);
      } else if(arg == "-ni") {
	args["ni"] = string(argv[++i]);
      } else if(arg == "-e") {
	args["epochs"] = string(argv[++i]);
      } else if(arg == "-s") {
	args["savedelta"] = string(argv[++i]);
      } else if(arg == "-d") {
	args["datasetdir"] = string(argv[++i]);
      } else if(arg == "-r") {
	args["random_net"] = string("1");
      } else if(arg == "-o") {
	args["onlinelearning"] = string("1");
      } else if(arg == "--train-set") {
	args["train_set"] = string(argv[++i]);
      } else if(arg == "--test-set") {
	args["test_set"] = string(argv[++i]);
      } else if(arg == "--validation-set") {
	args["validation_set"] = string(argv[++i]);
      } else if(arg == "--threshold-error") {
	args["threshold_error"] = string(argv[++i]);
      } else {
	cerr << "Unknown switch " << argv[i] << "\n";
	throw BadOptionSetting(usage);
      }
    }
  }

  _num_ops++;
  // Had to specify exactly one type of operation
  if(_num_ops == 0 || _num_ops > 1)
    throw BadOptionSetting(usage);
	  
}

Options* Options::instance() {
  if(_instance == 0) {
    string option_type(getenv("RNNOPTIONTYPE"));

    if(option_type == "train")
      _instance = new RNNTrainingOptions;
    // should probably exit with error, general options have no context
    else
      _instance = new Options;
  }

  return _instance;
}


// static member initialization
Options* Options::_instance = 0;

