#ifndef OPTIONS_H
#define OPTIONS_H

#include <map>
#include <vector>
#include <string>
#include <exception>

// Enum to set trasduction types flags
typedef enum TrasdType { 
  IOIS = 1, // IO-Isomorph structural trasduction
  SS, // Supersource trasduction
  IOIS_SS // Both the above types
} TrasdType;

/* 
  This class manage all application
  base options that are globally visible
*/

class Options {
/* SingleTon Pattern */
  static Options* _instance;
  
protected:
  // These are the global configuration options
  int _input_dimension, _domain_outdegree, _r, _s;
  std::vector<int> _lnunits;
  /*** UPDATE 27/01/2001 ***/
  std::vector<bool> _input_mask;
  TrasdType _trasd;
  bool _process_dr;
  int _precision; // float precision used for reading and saving net weights
  float _cthreshold; // contact threshold
  
  // This map stores all arguments value in the form of strings.
  // Clients have to convert to the appropriate type before using an argument
  std::map<std::string, std::string> args;

  /* SingleTon Pattern */
  
  // The constructor is not directly accessible
  // and initializes the map with default values.
  Options() {
    // Global configurations file: default to cwd
    args.insert(std::make_pair(std::string("conf_file"), std::string("./global.cnf")));
  }

public:
 
  class BadOptionSetting { // : public exception { // Why parse error?
    std::string _error_string;

  public:
      BadOptionSetting(std::string what): 
	      _error_string(what) {}

      const char* explainError() const {
        return _error_string.c_str();
      }
  };
  
  static Options* instance();
  
  virtual void parse_args(int argc, char* argv[])
    throw(BadOptionSetting);

  virtual std::string getUsageString() const { return ""; }

  std::string getParameter(std::string) const;

  void setParameter(std::string key, std::string value) {
    args.insert(make_pair(key, value));
  }

  // Global parameters accessor member functions
  int getInputDimension() const { return _input_dimension; }
  void setInputDimension(int n) { _input_dimension = n; }
  int getDomainOutDegree() const { return _domain_outdegree; }
  void setDomainOutDegree(int v) { _domain_outdegree = v; }
  std::pair<int, int> getLayersIndexes() const { return std::make_pair(_r, _s); }
  void setLayersIndexes(int r, int s) { _r = r; _s = s; }
  std::vector<int> getLayersUnitsNumber() const { return _lnunits; }
  void setLayersUnitsNumber(const std::vector<int>& lnunits) { _lnunits = lnunits; }
  TrasdType getTrasductionType() const { return _trasd; }
  void setTrasductionType(TrasdType trasd) { _trasd = trasd; }
  bool getProcessingFlag() const { return _process_dr; }
  int getPrecision() const { return _precision; }
  float getContactThreshold() const { return _cthreshold; }
  std::vector<bool> getInputMask() { return _input_mask; }
};

// We define classes derived from general Options
// class to manage application specific command line
// argument and general options

/*
  An option class to manage Recursive Neural Network
  training applications.
*/
class RNNTrainingOptions: public Options {
  // keeps track of number of user specified operations.
  // In general, it is an error if greater than 1.
  int _num_ops;
  std::string usage; // usage string to be filled with specific program name

public:
  
  RNNTrainingOptions() {
    // Default values for command line parameters
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
    usage = " [Options]\n"
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
      "       -r training start with random weights (default is read network from file\n"
      "       --train-set <file listing training set files in data directory>\n"
      "       --test-set <file listing test set files in data directory>\n"
      "       --validation-set <file listing validation set files in data directory>\n"
      "       --threshold-error <threshold error to be used to stop training> (default is 1e-3)\n";
      
  }											    
  void parse_args(int argc, char* argv[])
    throw(BadOptionSetting);

  virtual std::string getUsageString() const { return usage; }
};

/*
  An option class to manage applications for
  proteins graphs search.
*/
class ProteinSearchOptions: public Options {
  // keeps track of number of user specified operations.
  // In general, it is an error if greater than 1.
  int _num_ops;
  std::string usage; // usage string to be filled with specific program name

public:
  
  ProteinSearchOptions() {
    // Default values for command line parameters
    args.insert(std::make_pair(std::string("beamsize"), std::string("1")));
    //args.insert(std::make_pair(std::string("usestats"), std::string("1")));

    // Usage string: program name is added during command
    // line parsing.
    usage = " [Options]\n"
      "Options:\n"
      "       -c <global configurations file> (default is ./global.cnf)\n"
      "       -s <connection statistics file>"
      "       -n <network file>\n"
      "       -d <dssp data directory>\n"
      "       --data-set <file listing proteins files for which to make predictions>\n"
      //"       -f <file containing proteins definitions>\n"
      //"       -p <protein name to make prediction for>\n"
      "       -o <output file where to store prediction results>\n"
      "       -b <beam_size> (default is 1 --> Hill Climbing)\n";
  }											    
  void parse_args(int argc, char* argv[])
    throw(BadOptionSetting);

  virtual std::string getUsageString() const { return usage; }
};


#endif //OPTIONS_H
