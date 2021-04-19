# HMM
This is an implementation of a hidden Markov model (HMM) class in R I needed for a project to cluster movement data.

The forward algorithm is implemented in C++ using the RcppEigen library.

At the moment it provides gamma emission distributions only but I'm planning to generalize to arbitrary distributions in the future.

# Acknowledgements
The code is largely based on `Zucchini, MacDonald, and Langrock 2009 Hidden Markov Models for Time Series` as well as slides from Roland Langrock.
