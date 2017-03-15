#include "StructuredDomain.h"
#include "DPAG.h"
#include "Node.h"
#include <string>
#include <vector>

class InstanceParser;

/*
  
  An instance of a structured domain.

 */
class Instance {
  // instances might have an id, which makes it easier to
  // evaluate and report performance on an instance basis
  std::string _id;

  // the structural domain this instance belongs to
  Domain _domain;

  Transduction _transduction;
  bool _supervised;
  
  
  
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
    unsigned int _norient; // number of orientations
    DPAG** _orientations; // each orientation can be defined as a DPAG
    std::vector<int>* _top_orders;

  public:
    Skeleton(Domain);
    // TODO: copy constructor
    // Skeleton(const Skeleton&);
    ~Skeleton();

    int maximum_indegree() const { assert(_i>=0); return _i; }
    int maximum_outdegree() const { assert(_o>=0); return _o; }

    unsigned int num_orient() const { return _norient; }
    void orientation(unsigned int, DPAG*);
    const DPAG* orientation(unsigned int);
    DPAG** orientations() { return _orientations; }
    std::vector<int> topological_order(unsigned int) const;
    const std::vector<int>* topological_orders() { return _top_orders; }
  };

  Skeleton* _skel;

  friend class InstanceParser;
  
 public:
  
 Instance(Domain domain, Transduction transduction, bool supervised):
  _domain(domain), _transduction(transduction), _supervised(supervised), _skel(NULL) {}
  // TODO: copy constructor
  // Instance(const Instance&);
  virtual ~Instance() {
    for(std::vector<Node*>::iterator it=_nodes.begin(); it!=_nodes.end(); ++it) {
      delete *it;
      *it = 0;
    }

    delete _skel;
  }

  class BadInstanceCreation: public std::logic_error {
  public:
  BadInstanceCreation(std::string msg):
    logic_error(msg) {}
  };
  
  std::string id() const { return _id; }
  void id(const std::string& id) { _id = id; }
  Domain domain() const { return _domain; }
  Transduction transduction() const { return _transduction; }
  void skeleton(Skeleton* skel) {
    if(_skel != NULL) delete _skel;
    _skel = skel;
  }
  
  // super-source transduction, get/set structure target
  int output_dim() const { return _target.size(); }
  std::vector<float> target() { return _target; }
  void load_target(const std::vector<float>& target) { _target = target; }

  // though used internally, these are made public since other
  // actors might want to set node features, e.g. instance is
  // made on-the-fly for prediction
  Node* node(unsigned int n) { assert(n>0 && n<_nodes.size()); assert(_nodes[n] != NULL); return _nodes[n]; }
  void node(unsigned int n, Node* node) {
    assert(n>0 && n<_nodes.size());
    if(_nodes[n] != NULL)
      delete _nodes[n];
    _nodes[n] = node;
  }
  void load_input(unsigned int n, const std::vector<float>& input) { // load node n input
    assert(_nodes[n] != NULL);
    _nodes[n]->load_input(input);
  }
  void load_target(unsigned int n, const std::vector<float>& target) { // load node n target
    assert(_nodes[n] != NULL);
    _nodes[n]->load_target(target);
  }
  // TODO: these are temporary implementations, it depends on how the Node interface develops
  /* int node_input_dim() const { return _nodes[0]->_encodedeInput.size(); } */
  /* int node_output_dim() const { return _nodes[0]->_otargets.size(); } */

};
