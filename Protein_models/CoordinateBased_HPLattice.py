'''
HP-lattice Coordninate-based HP-lattice model
Based on Lucas Knuthsons code
'''

import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import math
from itertools import product
import matplotlib


class CoordinateBased_HPLattice:
	'''
	Class for one instance of the 2D lattice HP-model to then be fed into the quantum simulation.

	Class variables in order of creation [type]:
	After init:
		- dim_lattice = dimensions of the lattice (rows, columns) [tuple of two ints]
		- lambda_vector = penalty terms for the energy terms [tuple of three ints]
		- sequence = Given sequence of the problem. Sequence format: H is 1 and 0 is P, e.g., [H, P, P, H] = [1, 0, 0, 1]. [list of int/binaries]
		- Q = dict with energy info [dictionary]
		- bit_name_list = names of the qubits saved in a list. Name format: (node [tuple], seq [int]) [list of tuples]
		- num_bits = number of bits in the instance [int]
		- O_energies = one-body energies [list of floats]
		- T_energies = two-body energies [list of Numpy Arrays with floats]
		- Dn = Dn vector [number of nodes for placements for each amino acid in sequence] [list] DOES NOT WORK FOR RECTANGULAR LATTICES!!!

	After function call:
		- feasible_set = Numpy arrays of all feasible bitstrings solutions in one-hot encoding [list of Numpy Arrays]
		- solution_set = Numpy arrays of all bitstring solutions in one-hot encoding [list of Numpy Arrays]

	TODO:
	- comments

	'''

	def __init__(self, dim_lattice, sequence, lambda_vector = (1, 1, 1)):

		self.dim_lattice = dim_lattice
		self.lambda_vector = lambda_vector # best lambdas: lambda_1, lambda_2, lambda_3 = 1, 2, 1.5
		self.sequence = sequence
		self.Q = self.make_Q()

		self.bit_name_list = self.get_bit_names()
		self.num_bits = len(self.bit_name_list)
		self.O_energies = self.get_O_energies()
		self.T_energies = self.get_T_energies()
		self.Dn = self.get_Dn()

	def __str__(self):
		return '\nO:\n' + str(self.O_energies) + '\nT:\n' + str(self.T_energies) + '\nDn:\n' + str(self.Dn)

	def get_H_indices(self):
		'''
		The following lists are lists of the indices where
		Hs are positioned.
		Used in make_Q.
		Based on code by: Lucas Knuthson
		'''
		H_index_even = [i for i in range(len(self.sequence)) if self.sequence[i] == 1 and i % 2 == 0]
		H_index_odd = [i for i in range(len(self.sequence)) if self.sequence[i] == 1 and i % 2 == 1]
		return H_index_even, H_index_odd

	def combos_of_H(self):
		'''
		Used in make_Q.
		Based on code by: Lucas Knuthson
		'''
		H_index_even, H_index_odd = self.get_H_indices()
		H_combos = []
		for even in H_index_even:
			for odd in H_index_odd:
				H_combos.append((even, odd))
		return H_combos

	def split_evenodd(self):
		'''
		Split the sequence into a lists of odd and even beads.
		Cates = categories.
		Used in make_Q.
		Based on code by: Lucas Knuthson
		'''
		cates_even = [i for i in range(len(self.sequence)) if i%2 == 0]
		cates_odd = [i for i in range(len(self.sequence)) if i%2 == 1]
		return cates_even, cates_odd

	def make_Q(self, verbose = False):
		'''
		Q is the interactions in the ising model.
		Two-body energy: (q_1, q_2) = value
		One-body energy: (q_1, q_1) = value

		bit format: (node, seq. index)
		Node format: (row, col)
		Based on code by: Lucas Knuthson
		'''
		Q = defaultdict(int)

		cates_even, cates_odd = self.split_evenodd()

		G = nx.grid_2d_graph(self.dim_lattice[0], self.dim_lattice[1]) # makes a lattice as a graph

		# EPH
		H_combos = self.combos_of_H()
		# 4 ??
		count_HP = 0
		#print('HP')
		for u,v in G.edges():
			for x,y in H_combos:
				if (x-y)**2 > 4:
					if sum(u) % 2 != 1 and sum(v) % 2 == 1:
						Q[((u,x), (v,y))] -= 1
						count_HP += 1
						#print('-', ((u,x), (v,y)))
					elif sum(u) % 2 == 1 and sum(v) % 2 != 1:
						Q[((v,x), (u,y))] -= 1
						count_HP += 1
						#print('-', ((v,x), (u,y)))
					
		# Sums over the squared sums, lambda 1
		count_onper = 0
		#print('Sums over the squared sums')
		#even
		for i in cates_even:
			# One body
			for u in G.nodes():
				if sum(u) % 2 != 1:
					Q[((u,i), (u,i))] -= 1*self.lambda_vector[0]
					count_onper += 1
					#print('-', ((u,i), (u,i)))

			# Two body
			for u in G.nodes():
				for v in G.nodes():
					if u != v and (sum(u) % 2 != 1 and sum(v) % 2 != 1) :
						Q[((u,i),(v,i))] += 2*self.lambda_vector[0]
						count_onper += 1
						#print(((u,i),(v,i)))

		#odd
		for i in cates_odd:
			# One body
			for u in G.nodes():
				if sum(u) % 2 == 1:
					Q[((u,i),(u,i))] -= 1*self.lambda_vector[0]
					count_onper += 1
					#print('-', ((u,i),(u,i)))

			# Two body
			for u in G.nodes():
				for v in G.nodes():
					if u != v and (sum(u) % 2 == 1 and sum(v) % 2 == 1):
						Q[((u,i),(v,i))] += 2*self.lambda_vector[0]
						count_onper += 1
						#print(((u,i),(v,i)))

		# self-avoidance, lambda 2
		#print('self-avoidance')
		count_sa = 0
		for u in G.nodes():
			if sum(u) % 2 != 1: # even beads
				for x in cates_even:
					for y in cates_even:
						if x != y and x < y:
							Q[((u,x), (u,y))] += 1*self.lambda_vector[1]
							count_sa += 1
							#print(((u,x), (u,y)))
			elif sum(u) % 2 == 1: # odd beads
				for x in cates_odd:
					for y in cates_odd:
						if x != y and x < y:
							Q[((u,x), (u,y))] += 1*self.lambda_vector[1]
							count_sa += 1
							#print(((u,x), (u,y)))

		# Connectivity sums, lambda 3
		# Even
		#print('Connectivity sums')
		count_con = 0
		for i in cates_even:
			for u in G.nodes():
				for v in G.nodes():
					if (((u,v) in G.edges()) == False and ((v,u) in G.edges()) == False) and u != v:
						if sum(u) % 2 != 1 and sum(v) % 2 == 1 and len(self.sequence) % 2 == 0:
							count_con += 1
							#print(((u,i), (v,i+1)))
							Q[((u,i), (v,i+1))] += 1*self.lambda_vector[2]

						elif sum(u) % 2 != 1 and sum(v) % 2 == 1 and len(self.sequence) % 2 == 1:
							if i != cates_even[-1]:
								Q[((u,i), (v,i+1))] += 1*self.lambda_vector[2]
								count_con += 1
								#print(((u,i), (v,i+1)))
		# Odd
		for i in cates_odd:
			for u in G.nodes():
				for v in G.nodes():
					if (((u,v) in G.edges()) == False and ((v,u) in G.edges()) == False) and u != v:
						if (sum(u) % 2 != 1 and sum(v) % 2 == 1) and len(self.sequence) % 2 == 1:
							Q[((u,i+1), (v,i))] += 1*self.lambda_vector[2]
							count_con += 1
							#print(((u,i+1), (v,i)))

						elif (sum(u) % 2 != 1 and sum(v) % 2 == 1) and len(self.sequence) % 2 == 0:
							if i != cates_odd[-1]:
								count_con += 1
								#print(((u,i+1), (v,i)))
								Q[((u,i+1), (v,i))] += 1*self.lambda_vector[2]

		if verbose:
			print('Counts:')
			print('HP: ', count_HP)
			print('onper, lambda 1: ', count_onper)
			print('self-avoidance, lambda 2: ', count_sa)
			print('connectivity, lambda 3: ', count_con)

		Q = dict(Q) # not a defaultdict anymore to not be able to grow by error
		return Q

	def get_node_list(self, verbose = False):
		'''
		Returns a list of the nodes in the right order: snakey!
		Verbose will print resulting list and saves a .png of the graph.
		'''
		node_list = []
		(Lrow, Lcol) = self.dim_lattice
		G = nx.grid_2d_graph(Lrow, Lcol)

		for row in range(Lrow):
			start_index = row * Lcol
			if row % 2 == 0: # even row is forward
				node_list.extend(list(G.nodes())[start_index:start_index + Lcol])
			if row % 2 == 1: # odd row is backward
				node_list.extend(reversed(list(G.nodes())[start_index:start_index + Lcol]))
		if verbose:
			nx.draw(G, with_labels=True)
			plt.savefig('Lattice')
			print(node_list)
		return node_list

	def get_bit_names(self):
		'''
		Returns a list of all the bitnames in the form (node (row, col), seq) in the right order.
		'''

		seq_index = range(len(self.sequence))
		node_list = self.get_node_list(verbose = False)
		bit_name_list = []
		L_2 = int(self.dim_lattice[0]*self.dim_lattice[1])
		nodes_even = [x for x in range(L_2) if x % 2 == 0]
		nodes_odd = [x for x in range(L_2) if x % 2 != 0]
		# for all even nodes with first index aso
		for f in seq_index:
			if f % 2 == 0:
				for s in nodes_even:
					bit_name_list.append((node_list[s], f))
			if f % 2 == 1:
				for s in nodes_odd:
					bit_name_list.append((node_list[s], f))
		return bit_name_list

	def get_O_energies(self):
		O_energies = []
		for bit in self.bit_name_list:
			try:
				O_energies.append(self.Q[(bit, bit)])
			except:
				pass
		return O_energies

	def get_T_energies(self):
		'''
		Get the two-body energies for the Hamiltonian.
		'''
		T_energies = np.zeros((self.num_bits, self.num_bits))

		for j in range(self.num_bits):
			for k in range(self.num_bits):
				if j == k:
					T_energies[j,k] = 0
				else:
					try:
						T_energies[j,k] = self.Q[self.bit_name_list[j], self.bit_name_list[k]]
						if j > k:
							T_energies[k,j] = self.Q[self.bit_name_list[j], self.bit_name_list[k]]
					except:
						pass

		T_energies = np.triu(T_energies) # delete lower triangle
		T_energies = T_energies + T_energies.T - np.diag(np.diag(T_energies)) # copy upper triangle to lower triangle
		return T_energies

	def get_Dn(self):
		D = []
		for seq in range(len(self.sequence)):
			if seq % 2 == 0:
				D.append(math.ceil((self.dim_lattice[0]*self.dim_lattice[1])/2))
			if seq % 2 == 1:
				D.append(math.floor((self.dim_lattice[0]*self.dim_lattice[1])/2))
		return D

	def get_feasible_percentage(self):
		return 100*(len(self.feasible_set)/len(self.solution_set))

	def get_solution_set(self):
		'''
		Input: Number of bits.
		Output: Numpy arrays of dimensions (1, num_bits) in a list of all possible bitstrings.
		'''
		return [np.array(i) for i in product([0, 1], repeat = self.num_bits)]

	def get_feasible_set(self):
		'''
		Output: Numpy arrays of all feasible solutions, in a list.
		Hamming distance is 1 at each position.
		'''
		feasible_list = []
		index_list = []
		start = 0
		for rot in self.Dn:
			stop = start + rot
			index_perm = [x for x in range(start, stop)]
			index_list.append(index_perm)
			start = stop
		comb = list(product(*index_list))
		for i in comb:
			state = np.zeros(self.num_bits)
			for j in i:
				state[j] = 1

			feasible = True

			# same node and on?
			for b in range(self.num_bits):
				
				node1 = self.bit_name_list[b][0]
				node2 = self.bit_name_list[(b + self.dim_lattice[0]*self.dim_lattice[1]) % self.num_bits][0]
				if (node1 == node2) and state[b] and state[(b + self.dim_lattice[0]*self.dim_lattice[1]) % self.num_bits]:
					feasible = False
					break
			
			# longer distance than 1 manhattan distance
			if feasible:
				for bit1 in range(len(state)):
					found = False
					if state[bit1] == 1:
						for bit2 in range(bit1+1, len(state)):
							if state[bit2] == 1 and not found:
								found = True
								node1 = self.bit_name_list[bit1][0]
								node2 = self.bit_name_list[bit2][0]
								if self.manhattan_dist(node1, node2) > 1:
									feasible = False
									break
						else:
							continue
			if feasible:
				feasible_list.append(state)
		return feasible_list

	def manhattan_dist(self, node1, node2):
		distance = 0
		for node1_i, node2_i in zip(node1, node2):
			distance += abs(node1_i - node2_i)
		return int(distance)

	def calc_solution_sets(self):
		'''
		May take a while.
		'''
		self.feasible_set = self.get_feasible_set()
		self.solution_set = self.get_solution_set()

	def bit2energy(self, bit_array):
		Oe = np.dot(bit_array, self.O_energies)

		Te = 0
		for j,bit in enumerate(self.bit_name_list):
			for k,bit in enumerate(self.bit_name_list):
				if bit_array[j] == 1.0 and bit_array[k] == 1.0:
					Te += self.T_energies[j,k]
					
		energy = Oe + Te
		return energy

	def energy_of_set(self, feasible = False, verbose = False):
		energy_list = []
		labels = []
		mem = 1000000
		lowest_energy_bitstring = None
		if feasible:
			set_ = self.feasible_set
		else:
			set_ = self.solution_set
		for i in range(len(set_)):
			energy = self.bit2energy(set_[i])

			if verbose and (i%1000==0):
				print('Progress in energy calculations: ', round(100*i/len(set_), 1), '%%')
			try:
				energy = self.bit2energy(set_[i])
				if energy < mem:
					lowest_energy_bitstring = [set_[i], i, energy]
					mem = energy
				label = str(set_[i])
				label = label.replace(',', '')
				label = label.replace(' ', '')
				label = label.replace('.', '')
				labels.append(label)
			except:
				energy = 1000000
				if not feasible:
					label = str(set_[i])
					label = label.replace(',', '')
					label = label.replace(' ', '')
					label = label.replace('.', '')
					labels.append(label)
			energy_list.append(energy)
		print('Done!')
		return energy_list, labels, lowest_energy_bitstring

	def viz_solution_set(self, energy_for_set, labels, lowest_energy, title = '', sort = False):
		x = np.array(energy_for_set)

		if sort:
			sort_idx = x.argsort()
			x = x[sort_idx]
			labels = np.array(labels)[sort_idx]

		fig, ax = plt.subplots(figsize=(18, 4))
		plt.style.use("seaborn")
		matplotlib.rc('xtick', labelsize=12)
		ax.bar(range(len(energy_for_set)), x, tick_label = labels)

		if not sort:
			theoretical_lowest_idx = lowest_energy[1]
			ax.get_xticklabels()[theoretical_lowest_idx].set_color("green")
		else:
			ax.get_xticklabels()[0].set_color("green")

		plt.xlabel('Bitstrings')
		plt.xticks(rotation=85)
		plt.ylabel('Classic energy')
		plt.title(r'Classic energy for ' + title + ' bitstrings')

	def bit2coord(self, bit):
		x = []
		y = []
		for i in range(len(bit)):
			if int(bit[i]):
				x.append(self.bit_name_list[i][0][0])
				y.append(self.bit_name_list[i][0][1])
		return x, y

	def viz_lattice(self, bit):
		x_grid = range(self.dim_lattice[0])
		x_grid = [-x for x in x_grid]
		y_grid = range(self.dim_lattice[0])
		protein_grid = [0] * self.dim_lattice[0]

		plt.scatter(y_grid, x_grid, c=protein_grid, cmap='Greys', s=10)

		x, y = self.bit2coord(bit)
		#x = [0, 0, 1, 1]
		x = [-x for x in x]
		#y = [0, 1, 1, 0]
		
		plt.plot(y, x, 'k-', zorder=0)  # straight lines
		# large dots, set zorder=3 to draw the dots on top of the lines
		plt.scatter(y, x, c=self.sequence, cmap='coolwarm', s=1500, zorder=3) 

		plt.margins(0.2) # enough margin so that the large scatter dots don't touch the borders
		plt.gca().set_aspect('equal') # equal distances in x and y direction

		plt.axis('on')
		ax = plt.gca()
		ax.xaxis.set_major_locator(MultipleLocator(1))
		ax.xaxis.set_major_formatter(NullFormatter())
		ax.yaxis.set_major_locator(MultipleLocator(1))
		ax.yaxis.set_major_formatter(NullFormatter())
		ax.tick_params(axis='both', length=0)
		plt.grid(True, ls=':')
		plt.title(str(bit))

		for i in range(len(self.sequence)):
			plt.annotate(i, (y[i], x[i]), color='white', fontsize=24, weight='bold', ha='center')

