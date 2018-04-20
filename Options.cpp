#include "General.h"
#include "Options.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <fstream>
#include <string>
using namespace std;

void Options::parse_args(int argc, char* argv[]) 
  throw(BadOptionSetting) {

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
    throw BadOptionSetting("Must specify global configuration file; use -c <config file> command line flag");

  // read global configuration file and set globally visible parameters
  ifstream ifs((*it).second.c_str());
  if(!ifs)
    throw BadOptionSetting(string("Could not open ") + (*it).second.c_str() + "\n\n" + _usage);
  
  string line, dummy;
  while(getline(ifs, line)) {
    if(line[0] == '#') continue; // skip comments

    size_t pos;
    istringstream iss(line);
    
    pos = line.find("input_dimension");
    if(pos != string::npos) {
      iss >> dummy >> _input_dim;
      assert(_input_dim > 0);
      continue;
    }
    
    pos = line.find("output_dimension");
    if(pos != string::npos) {
      iss >> dummy >> _output_dim;
      assert(_output_dim > 0);
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
      _lnunits.clear();
      iss >> dummy >> _r >> _s;
      int i = 0, lnu;
      while(i<_r+_s) {
	iss >> lnu;
	_lnunits.push_back(lnu);
	++i;
      }
      assert(_lnunits.size() == (uint)_r+_s);
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
      else { throw BadOptionSetting("Unrecognised domain type"); }
      continue;
    }

    pos = line.find("transduction");
    if(pos != string::npos) {
      string trans;
      iss >> dummy >> trans;

      if(trans == "SUPER_SOURCE") { _transduction = SUPER_SOURCE; }
      else if(trans == "IO_ISOMORPH") { _transduction = IO_ISOMORPH; }
      else { throw BadOptionSetting("Transduction type not recognized"); }
      continue;
    }

    pos = line.find("problem");
    if(pos != string::npos) {
      string prob;
      iss >> dummy >> prob;

      if(prob == "REGRESSION") { _problem = REGRESSION; }
      else if(prob == "BINARYCLASS") { _problem = BINARYCLASS; }
      else if(prob == "MULTICLASS") { _problem = MULTICLASS; }
      else { throw BadOptionSetting("Problem type not recognized"); }
      continue;
    }

  }

  // raise exception if config parameters are not set
  if(!_input_dim)
    throw BadOptionSetting("Must set input_dimension to a positive value");
  if(!_output_dim)
    throw BadOptionSetting("Must set output_dimension to a positive value");
  if(!_domain_outdegree)
    throw BadOptionSetting("Must set domain_outdegree to a positive value");
  if(!_r)
    throw BadOptionSetting("Must set number of layers in folding network to a positive value");
  if(!_s)
    throw BadOptionSetting("Must set number of layers in transforming network to a positive value");
  if(!_lnunits.size())
    throw BadOptionSetting("Couldn't set architecture of the folding and tranforming networks");
  if(_transduction != SUPER_SOURCE && _transduction != IO_ISOMORPH)
    throw BadOptionSetting("Invalid transduction type: "  + _transduction);
  if(_problem & ~(REGRESSION | BINARYCLASS | MULTICLASS))
    throw BadOptionSetting("Invalid problem type");
}

void RNNTrainingOptions::parse_args(int argc, char* argv[]) 
  throw(Options::BadOptionSetting) {
  
  // parse configuration file first
  Options::parse_args(argc, argv);
 
  /*** parse command line and find application specific options ***/
  for (int i = 1; i < argc; i++) {
    // parse switches
    if (argv[i][0] == '-') {
      string arg(argv[i]);
      if(arg == "-c") {
	// this is already read by base object
	// TODO: should avoid considering it here
	++i;
      } else if(arg == "-n") {
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
      } else if(arg == "-r") {
	args["random_net"] = string("1");
      } else if(arg == "-o") {
	args["onlinelearning"] = string("1");
      } else if(arg == "--training-set") {
	args["training_set"] = string(argv[++i]);
      } else if(arg == "--test-set") {
	args["test_set"] = string(argv[++i]);
      } else if(arg == "--validation-set") {
	args["validation_set"] = string(argv[++i]);
      } else if(arg == "--threshold-error") {
	args["threshold_error"] = string(argv[++i]);
      } else {
	cerr << "Unknown switch " << argv[i] << "\n";
	throw BadOptionSetting(_usage);
      }
    }
  }
}

Options* Options::instance() throw(BadOptionSetting) {
  if(_instance == 0) {
    try {
      string option_type(getenv("RNNOPTIONTYPE"));
      if(option_type == "train")
	_instance = new RNNTrainingOptions;
      else
	throw BadOptionSetting("Invalid RNNOPTIONTYPE value");
    } catch (logic_error& e) {
      cerr << e.what() << endl;
      throw BadOptionSetting("Must set RNNOPTIONTYPE envirnoment variable");
    }
  }

  return _instance;
}

// static member initialization
Options* Options::_instance = 0;

