#include "General.h"
#include "require.h"
#include "Options.h"
#include "DataSet.h"
#include "RecursiveNN.h"
#include <cstdlib>
#include <cfloat>
#include <ctime>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

double adjustLearningRate(double curr_error, bool& restore_weights_flag, double& alpha) {
  /*** Vogl adaptive acceleration learning method (additive) ***/
  static double prev_eta = atof((Options::instance()->get_parameter("eta")).c_str());
  static double prev_error = FLT_MAX;

  double alpha_0 = 1e-1;
  double phi = 1.05, beta = .5, epsilon = 1e-2;
  double phi_additive = prev_eta/100;

  double curr_eta;

  if(curr_error < prev_error) {
    curr_eta = phi_additive + prev_eta;
    if(curr_eta > 1)
      curr_eta = 1;
    alpha = alpha_0;
    prev_eta = curr_eta;
    prev_error = curr_error;
  } else if(curr_error < (1 + epsilon) * prev_error) {
    curr_eta = beta * prev_eta;
    alpha = 0;
    prev_eta = curr_eta;
    prev_error = curr_error;
  } else {
    restore_weights_flag = true;
    curr_eta = beta * prev_eta;
    alpha = 0;
  }

  return curr_eta;
}

void computeErrorOnDataset();

void train(const string& netname, DataSet* trainingSet, DataSet* validationSet, ostream& os = cout) {
  // Get important training parameters
  bool onlinelearning = (atoi((Options::instance()->get_parameter("onlinelearning")).c_str()))?true:false;
  // bool random_net = (atoi((Options::instance()->get_parameter("random_net")).c_str()))?true:false;
  
  int epochs = atoi((Options::instance()->get_parameter("epochs")).c_str());
  int savedelta = atoi((Options::instance()->get_parameter("savedelta")).c_str());
  
  /*
   * read network in if it exists, otherwise make one from scratch
   */
  RecursiveNN<TanH, Sigmoid, MGradientDescent>* rnn;

  if(trainingSet->size()) {
    os << "Creating new network...";
    rnn = new RecursiveNN<TanH, Sigmoid, MGradientDescent>();
    os << " Done." << endl;
  } else {
    os << "Need some data to train network, please specify value for the --training-set argument\n";
    return;
  }

  os << endl << endl;
  
  bool restore_weights_flag = false;
  double curr_eta = atof((Options::instance()->get_parameter("eta")).c_str());
  double alpha = .9;
  
  double prev_error = FLT_MAX, min_error = FLT_MAX;
  int min_error_epoch = -1;
  double threshold_error = atof((Options::instance()->get_parameter("threshold_error")).c_str());

  for(int epoch = 1; epoch<=epochs; epoch++) {
    os << "Epoch " << epoch << '\t';
    
    for(DataSet::iterator it=trainingSet->begin(); it!=trainingSet->end(); ++it) {
      // (*it)->print(os);
      rnn->propagateStructuredInput(*it);
      rnn->backPropagateError(*it);

      /* stochastic (i.e. online) gradient descent */
      if(onlinelearning) {
      	if(restore_weights_flag)
      	  rnn->restorePrevWeights();
	
      	rnn->adjustWeights(curr_eta, alpha);
      	//curr_eta = adjustLearningRate(curr_train_error, restore_weights_flag, alpha);
      }
    }

    /* batch weight update */
    if(!onlinelearning) {
      if(restore_weights_flag)
    	rnn->restorePrevWeights();

      rnn->adjustWeights(curr_eta, alpha);
      //curr_eta = adjustLearningRate(curr_train_error, restore_weights_flag, alpha);
    }

    double error;
    double error_training_set = rnn->computeErrorOnDataset(trainingSet);
    os << "E_training = " << error_training_set << '\t';
    
    if(validationSet) {
      double error_validation_set = rnn->computeErrorOnDataset(validationSet);
      error = error_validation_set;
      os << "E_validation = " << error_validation_set;
    } else
      error = error_training_set;

    os << endl;

    if(min_error > error) {
      min_error = error;
      min_error_epoch = epoch;
      rnn->saveParameters(netname.c_str());
    }
    
    // stopping criterion based on error threshold
    if(fabs(prev_error - error) < threshold_error) {
      os << endl << endl << "Network error decay below given threshold. Stopping training..." << endl;
      break;
    }

    prev_error = error;

    // save network every 'savedelta' epochs
    if(!(epoch % savedelta)) {
      ostringstream oss;
      oss << netname << '.' << epoch;
      rnn->saveParameters((oss.str()).c_str());
    }
    
  }
  
  os << endl << flush;

  // deallocate Recursive Neural Network instace
  delete rnn; rnn = 0;

}

int main(int argc, char* argv[]) {
  setenv("RNNOPTIONTYPE", "train", 1);

  DataSet *trainingSet = NULL, *testSet = NULL, *validationSet = NULL;
  string netname;
  
  try {
    Options::instance()->parse_args(argc,argv);
    
    netname = Options::instance()->get_parameter("netname");
    if(!netname.length()) {
      cerr << "Must specify a network file" << endl << endl;
      throw Options::BadOptionSetting(Options::instance()->usage());
    }

    string training_set_fname = Options::instance()->get_parameter("training_set");

    if(training_set_fname.length()) {
      cout << "Creating training set. " << flush;
      trainingSet = new DataSet(training_set_fname.c_str());
      cout << "Done." << flush << endl;
    } else 
      cout << "Training set not specified. Skipping training..." << endl;

    string test_set_fname = Options::instance()->get_parameter("test_set");
    if(test_set_fname.length()) {
      cout << "Creating test set. " << flush;
      testSet = new DataSet(test_set_fname.c_str());
      cout << "Done." << flush << endl;
    } else 
      cout << "Test set not specified. Skipping testing..." << endl;

    string validation_set_fname = Options::instance()->get_parameter("validation_set");
    if(validation_set_fname.length()) {
      cout << "Creating validation set. " << flush;
      validationSet = new DataSet(validation_set_fname.c_str());
      cout << "Done." << flush << endl;
    }
  } catch(Options::BadOptionSetting e) {
    cerr << e.what() << endl;
    exit(EXIT_FAILURE);
  }
  
  /*** Train the network and save results ***/
  if(trainingSet) {
    cout << "Training set has " << (trainingSet->size()) << " instances." << endl;
    if(validationSet) {
      cout << "Training network with validation set." << endl;
      cout << "Validation set has " << validationSet->size() << " instances." << endl;
    }
    else
      cout << "Training without validation set." << endl;
      
    train(netname, trainingSet, validationSet);
    cout << "RNN model saved to file " << netname << endl;

    /* 
     * TODO 
     * predict(netname, trainingSet, "training.pred");
     * predict(netname, validationSet, "validation.pred");
     */

    delete trainingSet;
    if(validationSet)
      delete validationSet;
    
  } else if(validationSet) {
    cerr << "Cannot use validation set without training set. Skipping training..." << endl;
    delete validationSet;
  }

  if(testSet) {
    cout << "Test set has " << testSet->size() << " instances." << endl
	 << "Evaluating test set performance using network defined in file " << netname << endl;

    /*
     * TODO
     * predict(netname, testSet, "test.pred");
     */
    delete testSet;
  }

  return EXIT_SUCCESS;
}
