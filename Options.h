#ifndef OPTIONS_H
#define OPTIONS_H

/*

  Manage program options, from the command line and from
  a global configuration file

 */

#include "StructuredDomain.h"

#include <map>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

/* 

  This class manage all application base options that are globally visible

*/

class Options {
  // singleton pattern
  static Options* _instance;
  
 protected:
  // usage string to be filled with specific program name
  std::string _usage;
  
  // global configuration options
  int _input_dim, _output_dim, _domain_outdegree, _r, _s;
  std::vector<int> _lnunits;
  int _precision;
  Domain _domain;
  Transduction _transduction;
  
  // a map to store all arguments value in the form of strings.
  // clients have to convert to the appropriate type before using an argument
  std::map<std::string, std::string> args;

  // singleton pattern: the constructor is not directly
  // accessible and initializes the map with default values.
 Options() {
    // global configurations file: default to cwd
    args.insert(std::make_pair(std::string("config"), std::string(".rnnrc")));

    // default values for global configuration parameters
    _domain = DOAG;
    _transduction = SUPER_SOURCE;
    _precision = std::cout.precision();

    // the other values must be specified by the user
    _input_dim = _output_dim = _domain_outdegree = _r = _s = 0;
    _lnunits.clear();
    
    _usage = "[Options]\n"
      "Options:\n"
      "\t-c <config file> (default is .rnnrc)\n";
  }
  
 public:
  
  class BadOptionSetting: public std::logic_error { 
 public:
 BadOptionSetting(std::string what): logic_error(what) {}
 };

  // static method to return the singleton
 static Options* instance() throw(BadOptionSetting);

  // parse arguments, base class reads configuration file
  // derived classes add specialised command line options
  virtual void parse_args(int argc, char* argv[])
    throw(BadOptionSetting);

  std::string usage() const { return _usage; }
    
  // method to get/set parameters
  std::string get_parameter(std::string name) const {
    std::map<std::string, std::string>::const_iterator it = args.find(name);
    if(it != args.end())
      return (*it).second;
    return std::string("");
  }
  void set_parameter(std::string key, std::string value) {
    args.insert(make_pair(key, value));
  }

  // global parameters accessor member functions
  int input_dim() const { return _input_dim; }
  int output_dim() const { return _output_dim; }
  int domain_outdegree() const { return _domain_outdegree; }
  std::pair<int, int> layers_indices() const { return std::make_pair(_r, _s); }
  std::vector<int> layers_number_units() const { return _lnunits; }
  int precision() const { return _precision; }
  Domain domain() const { return _domain; }
  void domain(Domain d) { _domain = d; }
  Transduction transduction() const { return _transduction; }
  void transduction(Transduction t) { _transduction = t; }
 
};

// We define classes derived from general Options
// class to manage application specific command line
// arguments and general options

/*
  An option class to manage Recursive Neural Network
  training applications.
*/
class RNNTrainingOptions: public Options {
  // keeps track of number of user specified operations.
  // In general, it is an error if greater than 1.
  int _num_ops;

public:
  
 RNNTrainingOptions():Options() {
    // default values for command line parameters
    args.insert(std::make_pair(std::string("eta"), std::string("0.001")));
    args.insert(std::make_pair(std::string("alpha"), std::string("0.1")));
    args.insert(std::make_pair(std::string("ni"), std::string("0")));
    args.insert(std::make_pair(std::string("epochs"), std::string("1000")));
    args.insert(std::make_pair(std::string("savedelta"), std::string("100")));
    args.insert(std::make_pair(std::string("onlinelearning"), std::string("0")));
    args.insert(std::make_pair(std::string("random_net"), std::string("0")));
    args.insert(std::make_pair(std::string("threshold_error"), std::string("0.001")));
    
    // Usage string: program name is added during command
    // line parsing.
    _usage = "[Options]\n"
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
      "       --threshold-error <threshold error to be used to stop training> (default is 1e-3)\n";
      
  }											    
  void parse_args(int argc, char* argv[])
    throw(BadOptionSetting);
  
};

#endif //OPTIONS_H
