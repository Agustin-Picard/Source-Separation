# Source-Separation
Simple Python script for solving a simplified source separation problem developed as a Signal Processing course at IMT Atlantique

Implementation of a couple of algorithms of the simplified source separation problem:
x_1 = a11 * r_1 + a12 * r_2
x_2 = a21 * r_1 + a22 * r_2
where x_i are the measurements by the sensors and the r_i are the actual signals that
need to be separated.

In the first algorithm (ICA), the sources are separated by first a whitening of the set of signals
through an eigenvalue decomposition so as to orthogonalize the inputs and a subsequent minimization of
the Kurtosis function, which returns a set of separating vectors. With these vectors, a matrix is
formed and the signals are then mixed to return the separated sources.

In the second algorithm (SOBI), this separation is performed through an averaging of the windowed signal
correlation and subsequent eigenvalue decomposition to find the separating matrix.
