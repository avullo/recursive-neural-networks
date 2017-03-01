#ifndef _STRUCTURED_DOMAIN_H_
#define _STRUCTURED_DOMAIN_H_

#include "DPAG.h"
#include <vector>

/*

  Define aspects related to the representation of structured domains.
  
*/

/*
  List supported structured domain types:

  - DOAG: general DOAG
  - SEQUENCE, serial order sequence, i.e. left-to-right sequence
  - LINEARCHAIN, unordered chain, i.e. bidirectional sequence processing
  - NARYTREE, n-ary tree
  - UG, undirected graph, i.e. multiple-orientations processing, assume serial order defined on vertex indices
  - GRID2D, two-dimensional grid, no orientation

*/
typedef enum Domain {
  DOAG = 1, 
  SEQUENCE,
  LINEARCHAIN,
  NARYTREE,
  UG, 
  GRID2D, 
} Domain;

// Return the number of orientations the RNN must consider
// to process an instance in the given domain
int numOrientations(Domain domain) {
  switch(domain) {
  case DOAG: return 1;
  case SEQUENCE: return 1;
  case LINEARCHAIN: return 2;
  case NARYTREE: return 1;
  case UG: return 2;
  case GRID2D: return 4;
  default: // TODO raise exception
  }
}

/*
  
  Represents the skeleton of a data structure.

  Given a data structure, the skeleton is structure obtained
  by ignoring all node labels. The classical theory is restricted 
  to the domain of DOAGs. In order to deal with general undirected 
  structures, a skeleton object maintains a set of structures
  together with their topological orderings, one pair for every
  possible orientation that can be defined.
  
 */

class Skeleton {
  int _i; // max indegree
  int _o; // max outdegree

  /*
    cache for the set of possible oriented structures with their
    corresponding topological orderings
    
    NOTE: might implement lazy loading scheme
  */
  int _norient; // number of orientations
  DPAG* _orientations; // each orientation can be defined as a DPAG
  std::vector<int>* _top_orders;
 
 public:
  Skeleton(Domain);
  ~Skeleton();

  int maxIndegree() const { return _i; }
  int maxOutdegree() const { return _o; }

  // A skeleton is defined for a structured instance, which must
  // be able to set its internal parameters
  // forward declaration
  class StructuredInstance;
  friend class StructuredInstance;
};

#endif // _STRUCTURED_DOMAIN_H_
