# Variational Quantum Eigensolver
An implementation and guide for creating a variational quantum eigensolver (VQE).

This guide will focus on approximating the ground state of a hydrogen anion,
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
The device is created with the call to `qml.device`, and 
the number of electrons and provided to create the Hartree-Fock (HF)
state. The HF state can be viewed as an approximate ground state.
~~~python
dev = qml.device("lightning.qubit", wires=qubits)

electrons = 2
hf = qml.qchem.hf_state(electrons, qubits)
~~~
The parameters that we will be tuning are inputs to the Givens rotation.
In short, this is a unitary operation on the circuit that models the excitation of electrons.
PennyLane has a tutorial on the Givens rotation that can be found here: https://pennylane.ai/qml/demos/tutorial_givens_rotations.

We can then define the circuit with the possible Givens rotations when defining the `circuit`, and calculate 
the expectation value of the state with `cost_fn`.
~~~python
def circuit(theta, wires):
    #qml.BasisState(hf, wires=wires)
    qml.AllSinglesDoubles(weights=theta, wires=wires, hf_state=hf, singles=singles, doubles=doubles)

@qml.qnode(dev, interface="autograd")
def cost_fn(theta):
    circuit(theta, wires=range(qubits))
    return qml.expval(Hamiltonian)
~~~

# Optimization
The optimization of these parameters is done classically using gradient descent.
The code for this section is similar to the PennyLane demo found here: https://pennylane.ai/qml/demos/tutorial_vqe.
~~~python
energy = [cost_fn(theta)]
angle = [theta]

max_iterations = 100
conv_tol = 1e-06

for n in range(max_iterations):
    theta, prev_energy = opt.step_and_cost(cost_fn, theta)

    energy.append(cost_fn(theta))
    angle.append(theta)

    conv = np.abs(energy[-1] - prev_energy)

    if n % 2 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

    if conv <= conv_tol:
        break

print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
print("\n" f"Optimal value of the circuit parameter = {angle[-1]}")
~~~

After 19 iterations we converge to the final ground-state energy. `Final value of the ground-state energy = -0.46985398 Ha`.


