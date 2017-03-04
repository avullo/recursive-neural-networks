# recursive-neural-networks
Neural Networks for learning with structured data types. Recursive neural networks are a form of recurrent networks that is designed to process more generic data types than sequences, e.g. trees, DAGs, representing causal relationships among elements in a particular domain.

Learning is defined as the inference of approximations of binary mappings between input structured spaces and output structured spaces, called structural transductions. The inferred function can be used to solve either classification or regression problems defined on structures. 

The basic theory is described in [1], which defines the framework for learning deterministic causal transductions on the domain of ordered DAGs (include trees and linear sequences as special cases). The functions are learning by adjusting the weights of feed-forward neural networks by gradient descent using a generalised form of the well-known backpropagation propagation algorithm designed for structures, called Back-Propagaton Through Structure (BPTS). 

The software here implements an adaptation of recursive neural networks designed to process undirected graphs, possibly containing cycles. It allows learning of non-causal relationships over many possible structured domains, thus constituting a useful generalisation of the original theory. 

This software is a rewrite of the code I have implemented in support of my PhD and post-doc work while at the Univeristy of Firenze, University of California at Irvine and University College Dublin. It has been successfully applied in different problems related to structural bioinformatics, as described in [2,3,4].

# Desclaimer

This work aims to create a design and implementation of recursive neural networks that is better suited to be used by other people and integrated into external components, thus probably aspiring at becoming a library at some point.

It is work in progress and it's not usable at the moment. I am constantly adding new components until the design is complete. 

## References

[1] P. Frasconi, M. Gori and A. Sperduti. "A General Framework for Adaptive Processing of Data Structures", *IEEE Transactions on Neural Networks", 9:5, 768-786, 1998.

[2]

[3]

[4]