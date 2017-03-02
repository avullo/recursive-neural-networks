#include "require.h"
#include "Options.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iostream>
#include <string>
using namespace std;

#define PR(x) cout << #x << ": " << x << endl
#define WFP int z; cin >> z

void Options::parse_args(int argc, char* argv[]) 
  throw(Options::BadOptionSetting) {

  // Parse command line switches and search
  // for global configuration file option
  for(int i=1; i<argc; i++) {
    string option(argv[i]);
    
    if(option == "-c") {
      args["conf_file"] = string(argv[i+1]);
      continue;
    }
  }
  // In case required application parameters 
  // are not given echo usage message.
  map<string, string>::const_iterator it;
  
  it = args.find("conf_file");
  if(it == args.end())
    throw BadOptionSetting(string("Must specify global configuration file; use -c <conf_file> command line flag"));

  // Scan global configuration file and set globally visible parameters
  ifstream ifs((*it).second.c_str());
  assure(ifs, (*it).second.c_str());
  string line, dummy;
  while(getline(ifs, line)) {
    // Skip if line is a comment
    if(line[0] == '#')
      continue;
    int pos;
    istringstream iss(line);
    pos = line.find("input_dimension");
    if(pos != string::npos) {
      iss >> dummy >> _input_dimension;
      //PR(_input_dimension); //WFP;
      continue;
    }
    pos = line.find("domain_outdegree");
    if(pos != string::npos) {
      iss >> dummy >> _domain_outdegree;
      //PR(_domain_outdegree); //WFP;
      continue;
    }
    pos = line.find("contact_threshold");
    if(pos != string::npos) {
      iss >> dummy >> _cthreshold;
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
      //PR(_r); PR(_s);
      //copy(_lnunits.begin(), _lnunits.end(), ostream_iterator<int>(cout, " "));
      //cout << endl;
      continue;
    }
    pos = line.find("trasduction_type");
    if(pos != string::npos) {
      int temp;
      iss >> dummy >> temp;
      //PR(temp); //WFP;

      switch(temp) {
      case 1: _trasd = IOIS;
	break;
      case 2: _trasd = SS;
	break;
      case 3: _trasd = IOIS_SS;
	break;
      default:
	require(0, "Trasduction type not recognized");
      }
      continue;
    }
    pos = line.find("non_casual_processing");
    if(pos != string::npos) {
      iss >> dummy >> _process_dr;
      //PR(_process_dr); //WFP;
      continue;
    }
    pos = line.find("rnn_weights_precision");
    if(pos != string::npos) {
      iss >> dummy >> _precision;
      continue;
    }
    pos = line.find("inhibit_input_mask");
    if(pos != string::npos) {
      int i = 0;
      iss >> dummy;
      bool bit;
      while(i < _input_dimension) {
	iss >> bit;
	_input_mask.push_back(bit);
	++i;
      }
      if(_input_mask.size() != _input_dimension)
	throw BadOptionSetting(string("Inconsistency between input dim and input mask dim."));
      continue;
    }
  }
}

string Options::getParameter(string name) const { 
  map<string,string>::const_iterator it = args.find(name);
  if(it != args.end())
    return (*it).second;
  return string("");
}

void RNNTrainingOptions::parse_args(int argc, char* argv[]) 
  throw(Options::BadOptionSetting) {
  // Add program name to usage string
  usage = argv[0] + usage;

  // Scan for required general parameters
  try {
    Options::parse_args(argc, argv);
  } catch(Options::BadOptionSetting e) {
    cerr << e.explainError() << endl << endl;
    cerr << usage << endl;
    exit(-1);
  }
 
  /*** Scan command line and find application specific options ***/
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
      } else if(arg == "-c") {
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


void ProteinSearchOptions::parse_args(int argc, char* argv[]) 
  throw(Options::BadOptionSetting) {
  // Add program name to usage string
  usage = argv[0] + usage;

  // Scan for required general parameters
  try {
    Options::parse_args(argc, argv);
  } catch(Options::BadOptionSetting e) {
    cerr << e.explainError() << endl << endl;
    cerr << usage << endl;
    exit(-1);
  }
 
  /*** Scan command line and find application specific options ***/
  for (int i = 1; i < argc; i++) {
    // parse switches
    if (argv[i][0] == '-') {
      string arg(argv[i]);
      if(arg == "-n") {
	args["netname"] = string(argv[++i]);
      } else if(arg == "-s") {
	args["stats_file"] = string(argv[++i]);
      } else if(arg == "-d") {
	args["datasetdir"] = string(argv[++i]);
      } else if(arg == "--data-set") {
	args["data_set"] = string(argv[++i]);
      } //else if(arg == "-f") {
	//args["inputfile"] = string(argv[++i]);
      //} else if(arg == "-p") {
      //args["proteinname"] = string(argv[++i]);
      /*}*/ else if(arg == "-o") {
	args["outputfile"] = string(argv[++i]);
      } else if(arg == "-b") {
	args["beamsize"] = string(argv[++i]);
      } else if(arg == "-c") { // already read by parent class
      } else {
	/*** UPDATE 24/08/2001 ***/
	//cerr << "Unknown switch " << argv[i] << "\n";
	//throw BadOptionSetting(usage);
      }
    }
  }

  _num_ops++;
  // Had to specify exactly one type of operation
  if(_num_ops == 0 || _num_ops > 1)
    throw BadOptionSetting(usage);

  /*** UPDATE 24/08/2001 ***/
  // temporary solution to manage parameters for NoisySearch
  // Assume distribution parameters are the last two from the command line
  //args["param1"] = string(argv[argc-2]);
  //args["param2"] = string(argv[argc-1]);
}


Options* Options::instance() {
  if(_instance == 0) {
    string option_type(getenv("OPTIONTYPE"));

    if(option_type == "RNNTrainingOptions")
      _instance = new RNNTrainingOptions;
    else if(option_type == "ProteinSearchOptions")
      _instance = new ProteinSearchOptions;
    else
      _instance = new Options;
  }

  return _instance;
}


// Static member initialization
Options* Options::_instance = 0;

