from pennylane import numpy as np
import pennylane as qml

def protein_folding_hamiltonian(ProteinInstance):
	'''
	Input: Protein instance from CoordinateBased_HPLattice or RotamerRosetta.
	Returns: The hamiltonian for the protein instance.
	'''
	num_qubits = len(ProteinInstance.O_energies)
	wires = range(ProteinInstance.num_bits)

	O_coeffs = [-x/2 for x in ProteinInstance.O_energies] 

	T_coeffs = np.copy(ProteinInstance.T_energies)

	for j in range(num_qubits):                         
		for k in range(num_qubits):
			T_coeffs[j,k] = T_coeffs[j,k]/4

	H_cost = get_cost_hamiltonian(O_coeffs, T_coeffs, wires)
	return H_cost

def get_cost_hamiltonian(O_coeffs, T_coeffs, wires):
	H_cost_O = get_O_hamiltonian(O_coeffs, wires)
	H_cost_T = get_T_hamiltonian(T_coeffs, wires)
	return H_cost_O + H_cost_T

def get_O_hamiltonian(O_coeffs, wires):
	return qml.Hamiltonian(O_coeffs, [qml.PauliZ(i) for i in wires])

def get_T_hamiltonian(T_coeffs, wires):
	obs = []
	coeffs = []
	for j in wires:
		for k in range(j+1, len(wires)):
			coeffs.append(T_coeffs[j,k]) 
			coeffs.append(-T_coeffs[j,k])
			coeffs.append(-T_coeffs[j,k])
			obs.append(qml.PauliZ(j) @ qml.PauliZ(k))
			obs.append(qml.PauliZ(j))
			obs.append(qml.PauliZ(k))
	return qml.Hamiltonian(coeffs, obs)


