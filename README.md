This repository provides the code and additional simulation results corresponding to the paper [Generative neural networks for characteristic functions](https://arxiv.org/abs/2401.04778). 

# Run your own example
To run the code for your own example of a characteristic function you need your own (vectorized) implementation of the characteristic function in PyTorch, which allows to do the calculations fully on either CPU or GPU. 
Then simply copy one of the main_ files and replace the charfct and charfct_cpu function with your own implementation. The code should then run with the same hyperparameters as we used in the simulation study of the paper.
For visualizations one can consult the graphs_ files to reproduce plots/tables.

# Technical requirements
A spec file can be found in spec-file.txt. It contains a list of packages that was used in the implementation. It is likely that access to a GPU is required to be able to conduct the training in a resonable amount of time.
As an example, for a single fixed combination of hyperparameters as in the simulation study of the paper, the training of one model took about 1-1.5h on a GPU.
