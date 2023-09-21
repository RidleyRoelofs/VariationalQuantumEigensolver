# Variational Quantum Eigensolver
An implementation and guide for creating a variational quantum eigensolver (VQE).

This guide will focus approximating the ground state of a hydrogen anion,
also known as hydride. The PennyLane website provides many
guides similar to this one on their demos page here: https://pennylane.ai/qml/demonstrations

## VQE
In the following demonstration of code, we will try and 
model the molecular properties of a hydrogen anion 
using a quantum circuit with variable parameters. 
These parameters will be tuned in order to try and 
minimize the cost function, which in this case is the energy
of a molecule in a given state. By determining the parameters
that minimize the cost function, we will have found an
estimation for the ground state of the molecule.

## Basis Sets
The basis set is chosen as a way to mathematically represent 
the atomic orbital of a molecule. The two choices we consider
are the sto-3g, and cc-pvdz basis. The sto-3g basis is much 
less expensive to compute with, at the cost of accuracy.
We will see that for the case of hydride, a better basis
set approximation is needed and the cc-pvdz set provides
a suitable approximation.

# Building the Circuit
## Hamiltonian Creation
To begin, we will create a hamiltonian of our molecule using
qchem.molecular_hamiltonian. Recall that a hydride molecule 
consists of a hydrogen atom and two electrons. The single hydrogen 
is listed within the `symbols` variable. The `coordinates` variable holds the position
of each symbol in 3D-space. For simplicity the hydrogen atom is located at `x=0, y=0, z=0`.
The charge of `-1` is due to there being a net charge of `-2 + 1 = -1` from the 
two electrons and one proton. The cc-pvdz basis set was chosen for its increased accuracy that is needed
when evaluating the cost function.
~~~python
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np

symbols = ["H"]
coordinates = np.array([0.0, 0.0, 0.0])

Hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=-1, basis="cc-pvdz")
~~~

To view the hamiltonian and number of qubits needed, simply add some print statements

~~~python
print("# of qubits needed: ", qubits)
print(Hamiltonian)
~~~
## Circuit Creation

dev = qml.device("lightning.qubit", wires=qubits)
