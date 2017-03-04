# recursive-neural-networks
Neural Networks for learning with structured data types. Recursive neural networks are a form of recurrent networks that is designed to process more generic data types than sequences, e.g. trees, DAGs, representing causal relationships among elements in a particular domain.

Learning is defined as the inference of approximations of binary mappings between input structured spaces and output structured spaces, called structural transductions. The inferred function can be used to solve either classification or regression problems defined on structures. 

The basic theory is described in [6], which defines the framework for learning deterministic causal transductions on the domain of ordered DAGs (include trees and linear sequences as special cases). The functions are learning by adjusting the weights of feed-forward neural networks by gradient descent using a generalised form of the well-known backpropagation propagation algorithm designed for structures, called Back-Propagaton Through Structure (BPTS). 

The software here implements an adaptation of recursive neural networks designed to process undirected graphs, possibly containing cycles. It allows learning of non-causal relationships over many possible structured domains, thus constituting a useful generalisation of the original theory. 

This software is a rewrite of the code I have implemented in support of my PhD and post-doc work while at the Univeristy of Firenze, University of California at Irvine and University College Dublin. It has been successfully applied in different problems related to structural bioinformatics, as described in many pulications like e.g. [1-5].

# Disclaimer
This work aims to create a design and implementation of recursive neural networks that is better suited to be used by other people and integrated into external components, thus probably aspiring at becoming a library at some point. The moment I write I feel a little bit ambitiuous and dare to say it might even have bindings in some other language as well. 

This at the moment has more of an educational purpose, both personal as I practice some useful techniques like refactoring, design patterns, TDD, and aimed at people willing to learn and use about this kind of learning approaches.

It is work in progress and it's not usable at the moment. I am constantly adding new components until the design is complete. 

# Dependencies
This code relies upon the Boost Graph Library and the CATCH unit testing framework. The latter is included as third party component.

# Installation
The makefile provided is tailored to Linux environments with the GNU C++ compiler suite. It relies on the standard installation path of the Boost library headers.

## References

[1] Pollastri G., Vullo A., Frasconi P. and Baldi P.. "Modular DAG-RNN Architectures for Assembling Coarse Protein Structures", *Journal of Computational Biology*, 13(3), 2006

[2] Ceroni A., Passerini A., Vullo A. and Frasconi P.. "DISULFIND: a disulfide bonding state and cystein connectity server". , *Nucleic Acid Research*, 34(2), W177-W181, 2006.

[3] Baldi P., Cheng J., Vullo A. "Large-scale prediction of disulphide bond connectivity". In: Saul L.K., Weiss Y., Bottou L., editors. *Advances in Neural Information Processing Systems* 17; Cambridge, MA: MIT Press; 97–104, 2005.

[4] Vullo A. and Frasconi P.. "Disulfide Connectivity Prediction Using Recursive Neural Networks and Evolutionary Information", *Bioinformatics*, 20(5), 653-659, 2004. 

[5] Pollastri G., Baldi P., Vullo A. and Frasconi P.. "Prediction of Protein Topologies Using GIOHMMs and GRNNs", *Advances in Neural Information Processing Systems (NIPS)* 15, MIT Press, 2003.

[6] Frasconi P., Gori M. and Sperduti A.. "A General Framework for Adaptive Processing of Data Structures", *IEEE Transactions on Neural Networks*, 9:5, 768-786, 1998.
