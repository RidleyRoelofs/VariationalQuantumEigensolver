import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np

# Specify the symbol set and coordinates for Hydride
symbols = ["H"]
coordinates = np.array([0.0, 0.0, 0.0])

# Create the hamiltonian according to the specified basis set
Hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=-1, basis="cc-pvdz")

print("# of qubits needed: ", qubits)
print(Hamiltonian)

dev = qml.device("lightning.qubit", wires=qubits)

electrons = 2
hf = qml.qchem.hf_state(electrons, qubits)
print(hf)

theta = [np.array([0.0], requires_grad=True)]
singles, doubles = qchem.excitations(electrons=electrons, orbitals=qubits)
num_theta = len(singles) + len(doubles)

def circuit(theta, wires):
    qml.AllSinglesDoubles(weights=theta, wires=wires, hf_state=hf, singles=singles, doubles=doubles)

@qml.qnode(dev, interface="autograd")
def cost_fn(theta):
    circuit(theta, wires=range(qubits))
    return qml.expval(Hamiltonian)

stepsize = 0.4
max_iterations = 30
opt = qml.GradientDescentOptimizer(stepsize=stepsize)
theta = np.zeros(num_theta, requires_grad=True)


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








