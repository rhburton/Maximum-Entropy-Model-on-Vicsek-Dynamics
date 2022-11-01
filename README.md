# Maximum-Entropy-Model-on-Vicsek-Dynamics
This repository holds code used for my undergraduate senior thesis, along with a pdf of the thesis itself. The project was to build a maximum entropy model (MEM) over a simulated flock obeying Vicsek dynamics (a type of computational model used to study collective motion, see section 3 of the thesis or https://arxiv.org/pdf/cond-mat/0611743.pdf). The MEM itself is borrowed from work done by Bialek et al (section 4 or https://www.princeton.edu/~wbialek/our_papers/bialek+al_12.pdf) used on actual flocks of starlings, testing ideas like what kind of interaction rules the birds follow.

FlockSym.py saves snapshots of these Vicsek flocks to a text file, which is then read by MEMAnalysis.py to build the maximum entropy model. Multiple noise types and interaction types are supported in FlockSym.py, although only uniform noise with symmetric topological neighborhood rules were used in the paper.
