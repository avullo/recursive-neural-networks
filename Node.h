#ifndef _NODE_H
#define _NODE_H

#include "Options.h"
#include <vector>

/* 
   Manage DPAG node informations suitable to be processed
   by a recursive neural network.
*/

class Node {
  /* // Whether or not to process outputs in b layers */
/*   bool _process_dr; */
/*   // Wheter or not to do io-isomorf trasduction computations */
/*   bool _ios_tr; */
  
  // Allocation specific routines
  void allocFoldingOutputStruct(const std::vector<int>&, int, double***, double**);
  void allocHoutputStruct(const std::vector<int>&, int, int);

  // Deallocation specific routines
  void deallocFoldingOutputStruct(const std::vector<int>&, int, double***, double**);
  void deallocHoutputStruct(const std::vector<int>&, int);

  // Prevent Assignment
  Node& operator=(const Node&);
 public:
  /* // Useful quantities */
/*   int _v; // Valence of the domain (max outdegree of a node) */
/*   int _r; // Repres. layer (its number of units, m, indicates dimension of a substructure enconding */
/*   int _s; // Number of layers of h map. */

  // A simple vector to map symbolic
  // label of a node to numeric codes
  std::vector<float> _encodedInput;
  
  // Vector of target output label, in case
  // we implement an io-iosomorf trasduction
  std::vector<float> _otargets;

  // Store number of units per folding layer. It is responsibility
  // of pattern costruction routine to synchronize this data
  // with that of the recursive neural network that process it.
  /* std::vector<int> _lnunits; */

  /*
    Store ouput activations for each layer and for each folding part.
    This implements the copy of every layer of the folding
    part for every node of the structure processed (Goller §3.3.1),
    because we need to store activations of every layer for each node,
    and not the copy of weights of the unfolding part.
  */
  double** _f_layers_activations;
  double** _b_layers_activations;
  double** _j_layers_activations;
  double** _k_layers_activations;
  double** _h_layers_activations;

  /*
    Store delta values at representation layer for each node, so
    if a node have different immediate predecessors, its
    delta values coming from predecessors can be summed up.
    As reported in (Goller §3.3.2), during the computation
    of delta values for each layer, their contribution to the
    weight update can be computed (accumulated), so we do not need
    to store them (except those from representation layer, as already observed).
  */
  double* _f_delta_lr;
  double* _b_delta_lr;
  double* _j_delta_lr;
  double* _k_delta_lr;

  /* Constructors */
  Node() {}
  Node(const std::vector<float>&);

  // Must furnish copy constructor to safely 
  // build StructuredInstanceTemplate Node vector
  Node(const Node&);

  /* Destructor */
  ~Node();
  
  /*
    This method reset activation values in all layers 
    and delta values in representation so we can process 
    this node as another node of another structure.
  */
  void allocStructs(const std::vector<int>&, int, int, bool, bool);
  void deallocStructs(const std::vector<int>&, int, int, bool, bool);
  void resetValues();

  std::vector<float> input() { return _encodedInput; }
  void load_input(const std::vector<float>& input) { _encodedInput = input; }
  std::vector<float> target() { return _otargets; }
  void load_target(const std::vector<float>& otargets) { _otargets = otargets; };
};

#endif // _NODE_H
