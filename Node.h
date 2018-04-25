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

#ifndef _NODE_H
#define _NODE_H

#include <vector>

/* 
   Manage DPAG node information suitable to be processed
   by a recursive neural network.
*/

class Node {
  /*
    _r: the number of layers of the NNs implementing the state transition functions
    _s: the number of layers of the NN implementing the node output function
   */
  int _r, _s;
  
  // the number of units in each layer of the state transition and node output functions
  std::vector<int> _lnunits; 
  bool _ios_tr; // whether learning is for an IO-ISOMORPH transduction
  int _norient; // the number of data structure possible orientation, i.e. number of state transition functions
  
  // Allocation specific routines
  void allocFoldingOutputStruct(double***, double**);
  void allocHoutputStruct();

  // Deallocation specific routines
  void deallocFoldingOutputStruct(double***, double**);
  void deallocHoutputStruct();

  // Prevent Assignment
  Node& operator=(const Node&);
 public:

  // A simple vector to map symbolic
  // label of a node to numeric codes
  std::vector<float> _encodedInput;
  
  // Vector of target output label, in case
  // we implement an io-iosomorf trasduction
  std::vector<float> _otargets;
  std::vector<float> _outputs;

  /*
    Store ouput activations for each layer and for each folding part.
    This implements the copy of every layer of the folding
    part for every node of the structure processed (Goller §3.3.1),
    because we need to store activations of every layer for each node,
    and not the copy of weights of the unfolding part.
  */
  double*** _layers_activations;
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
  double** _delta_lr;

  /* Constructors */
  //Node(const std::vector<float>&);
  Node();

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
  // void allocStructs(const std::vector<int>&, int, int, bool, bool);
  // void deallocStructs(const std::vector<int>&, int, int, bool, bool);
  void resetValues();

  std::vector<float> input() { return _encodedInput; }
  int input_dim() const { return _encodedInput.size(); }
  void load_input(const std::vector<float>& input) { _encodedInput = input; }

  std::vector<float> target() { return _otargets; }
  std::vector<float> output() { return _outputs; }
  int output_dim() const { return _otargets.size(); }
  void load_target(const std::vector<float>& otargets) { _otargets = otargets; }
  void load_output(const std::vector<float>& outputs) { _outputs = outputs; }
};

#endif // _NODE_H
