#ifndef _STRUCTURED_DOMAIN_H_
#define _STRUCTURED_DOMAIN_H_

#include <stdexcept>

/*
 *
 * Define aspects related to the representation of structured domains.
 *
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
  DOAG = 0, 
  SEQUENCE,
  LINEARCHAIN,
  NARYTREE,
  UG, 
  GRID2D, 
} Domain;

// Return the number of orientations the RNN must consider
// to process an instance in the given domain
int num_orientations(Domain domain)
  throw(std::logic_error);
  
typedef enum Transduction {
  SUPER_SOURCE = 0,
  IO_ISOMORPH
} Transduction;

/*
 * Types of learning problems on a structured domain
 */
typedef enum {
  UNDEFINED   = 1<<0,
  REGRESSION  = 1<<1,
  BINARYCLASS = 1<<2,
  MULTICLASS  = 1<<3,
} Problem;

#endif // _STRUCTURED_DOMAIN_H_
