#ifndef _STRUCTURED_INSTANCE_H_
#define _STRUCTURED_INSTANCE_H_

/*
  An instance in a domain of labelled structures.
  
  It has inputs at nodes and outputs at either the level
  of nodes (i.e. learning IO-isomorph transductions) or
  at the level of the entire structure (i.e. super-source 
  transduction where only a single source node with target
  is defined).

  It might or might not have an ID associated to facilitate
  associations of predictions at the level of the entire
  structure.

  NOTE: support for processing of undirected structures, i.e.
        non-causal transductions
  
  The traditional case, i.e. learning over directed structures
  where just one orientation is defined, is subsumed and it  
  defaults to that one.

 */

#include "StructuredDomain.h"
#include "Node.h"
#include "DPAG.h"
//#include "TargetFunctions.h"

#include <string>
#include <vector>
#include <iostream>

class StructuredInstance {
  std::string _id; // the ID of the structure, optional

  // instance belongs to a given domain, which allows dynamically
  // setting the number of possible orientations to consider
  Domain _domain; 
  
  // the set  of underlying nodes is invariant wrt possible
  // orientations, maintains one unique copy of the nodes.
  std::vector<Node*> _nodes;

  // if super-source transduction, maintains the target associated to
  // the whole instance, not the nodes
  //Target* _structure_target; 

  // the skeleton of the instance, to cache the set of
  // possible structures defined by different orientations
  // with their topological orderings
  Skeleton _skel;
  
 public:

  // read (optional: read IDs), and initialise instance from input stream
  // NOTE: might want to use factory method
  StructuredInstance(std::istream& is, Domain = DOAG, bool = false);
  ~StructuredInstance() {
    for(std::vector<Node*>::iterator it = _nodes.begin(); it!=_nodes.end(); ++it)
      delete *it;
  }

  int numOrientations() const { return _skel._norient; }
  const DPAG* const skeleton(int i) const { return &(_skel.orientations[i]); }
  std::vector<int> topologicalOrder(int i) const { return _skel._top_orders[i]; }
  
  // serialize structure to output stream
  friend ostream& operator<<(std::ostream, const StructuredInstance&);
  
};

#endif // _STRUCTURED_INSTANCE_H_
