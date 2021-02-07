# Enigma

## Inspiration
As an airline, planning flights is critical for success and return on investment. 

Unfortunately, these plans change rapidly in day to day operations, as it’s difficult to account for what may go wrong.

One challenge is the fact many flights may be close to overlapping in schedule and location during operating hours which increases the risk of delays affecting other aircraft.  

What can we do to make our planning better? 

### AirLine DataSets
Data sets from the Air line industry with respect to flight routes
![cap5](../figures/)
### Trip Annealer 
Encodes graph Traveling Sales Problem inot QUBO on Dwave System

https://github.com/Enigma-CDL/Enigma/blob/master/TripGenerator.ipynb

https://github.com/Enigma-CDL/Enigma/blob/master/TripAnneal.ipynb

https://github.com/Enigma-CDL/Enigma/blob/master/SegmentsLoader.ipynb

### Classical GAN
A Classical Generative Adversarial Network designed to output adjacency tables that correlate the flight paths. The model is trained on optimized outputs generated by the D-Wave machine.

https://github.com/Enigma-CDL/Enigma/blob/master/Classical_Gan/Classical_GAN.ipynb

### QGan
The QGAN takes as its input a sparse matrix describing an adjacency matrix and a solution to the TSP problem as found by the D-Wave device. It will then build two shallow quantum circuits. The first will present examples to the second, with the aim of tricking the second into labeling a generated matrix as a real one. 

The second attempts to discriminate between generated matrices and real. Each is trained in parallel, and at the end of the training, we hope to have a generator that can generate good samples of solved TSP problems on our example graph. The structure of the total circuit differs if the algorithm is to be run on a continuous variable (qumode) or discrete (qubit) systems. But the ability to train for either system is built-in.

https://github.com/Enigma-CDL/Enigma/blob/master/qGAN/train_qGAN.ipynb

### Gaussian Boson Sampling
Gaussian Boson Sampling is employed in the continuous mode quantum computer from Xanadu. We use this platform's ability to sample from complex probability distributions to find dense sub-graphs within our total scheduling graph. This allows us to avoid these routes if possible, as the existence of a dense subgraph implies the existence of a congested route/area. This congestion increases operational risk.

https://github.com/Enigma-CDL/Enigma/blob/master/gbsGraphSelector.ipynb


