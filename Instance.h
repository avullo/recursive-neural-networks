#include "StructuredDomain.h"
#include "DPAG.h"
#include <string>
#include <vector>

/*
  
  An instance of a structured domain.

 */
class Instance {
 protected:
  // instances might have an id, which makes it easier to
  // evaluate and report performance on an instance basis
  std::string _id;

  // the structural domain this instance belongs to
  Domain _domain;
  Transduction _transduction;
  
  // learning a super-source transduction,
  // maintains the target associated to the whole structure
  std::vector<float> _target;
  
  // the list of nodes in the structure, indexed by their index in the graph
  std::vector<Node*> _nodes;

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
    DPAG** _orientations; // each orientation can be defined as a DPAG
    std::vector<int>* _top_orders;
    
  public:
    Skeleton(Domain);
    ~Skeleton();

    int max_indegree() const { assert(_i>=0); return _i; }
    int max_outdegree() const { assert(_o>=0); return _o; }

    // a skeleton is defined for a structured instance, which must
    // be able to set its internal parameters
    // forward declaration
    class Instance;
    friend class Instance;
  };

  Skeleton _skel;
  
 public:
  
 Instance(Doman domain, Transduction transduction):
  _domain(domain), _transduction(transduction), _skel(domain) {}
  virtual ~Instance() {
    for(vector<Node*>::iterator it=_nodes.begin(); it!=_nodes.end(); ++it)
      delete *it;
  }

  // Used with factory method
  class BadInstanceCreation: public std::logic_error {
  pulic:
  BadInstanceCreation(std::string type):
    logic_error("Cannot create type " + type) {}
  };

  // factory method
  static Instance* factory(Enum)
    throw(BadInstanceCreation);
  
  virtual void read(std::istream&, bool = true /* read target */) = 0;
  
  std::string id() const { return _id; }
  
  Domain domain() const { return _domain; }
  Transduction transduction() const { return _transduction; }

  // super-source transduction, load structure target
  void load_target(const std::vector<float>& target) { _target = target; }
  
  void load_input(int n, const std::vector<float>& input) { // load node n input
    assert(_nodes[n] != NULL);
    _nodes[n]->load_input(input);
  }
  void load_target(int n, const std::vector<float>& target) { // load node n target
    assert(_nodes[n] != NULL);
    _nodes[n]->load_target(target);
  }
  
};


class DOAG: public Instance {
 DOAG(Domain d, Transductiont t): Instance(d, t) {}
  friend class Instance;
  
 public:
  void read(std::istream, bool = true);
};

class Sequence: public Instance {
  Sequence(Domain d, Transductiont t): Instance(d, t) {}
  friend class Instance;
  
 public:
  void read(std::istream, bool = true);
};

class LinearChain: public Instance {
  LinearChain(Domain d, Transductiont t): Instance(d, t) {}
  friend class Instance;
  
 public:
  void read(std::istream, bool = true);
};

class UndirectedGraph: public Instance {
  UndirectedGraph(Domain d, Transductiont t): Instance(d, t) {}
  friend class Instance;
  
 public:
  void read(std::istream, bool = true);
};

class Grid2D: public Instance {
  Grid2D(Domain d, Transductiont t): Instance(d, t) {}
  friend class Instance;
  
 public:
  void read(std::istream, bool = true);
};
