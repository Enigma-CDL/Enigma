import tensorflow as tf
import pennylane as qml
import numpy as np

from typing import Union, List, Tuple


class qGAN:

    def __init__(self, n_qubits, gen_dev: str = 'default.qubit.tf',  disc_dev: str = 'default.qubit.tf'):
        self.n_qubits = n_qubits
        self.gen_dev = qml.device(gen_dev, wires=n_qubits)
        self.disc_dev = qml.device(disc_dev, wires=n_qubits)

    @staticmethod
    def tsp_cost(adjacency_matrix: np.ndarray, solution_vector: np.ndarray):
        order = np.where(solution_vector == 1)
        norm = np.sum(adjacency_matrix)
        adjacency_matrix = adjacency_matrix / norm
        last = order[0]
        cost = 0
        for next_node in order:
            cost += adjacency_matrix[next_node, last]
            last = next_node
        return cost

    @staticmethod
    def iSWAP(weight: Tuple[float], wires=List[int]):
        c = wires[0]
        t = wires[1]
        rot = weight
        qml.CNOT(wires=[c, t])
        qml.Hadamard(wires=c)
        qml.CNOT(wires=[t, c])
        qml.RZ(rot / 2, wires=c)
        qml.CNOT(wires=[t, c])
        qml.RZ(-rot / 2, wires=c)
        qml.Hadamard(wires=c)
        qml.CNOT(wires=[c, t])

    @staticmethod
    def time_ordered_to_adjacency(time_ordered: Union[np.ndarray, List[int]]):
        """
        Takes a matrix which is defined where each row describes a city and each column a time step, and turns these
        into a directed adjacency matrix - i.e. the row describes the starting node and the column the end node.
        :param time_ordered: A matrix or vector describing the solution to a TSP problem.
        :return: adjacency_matrix: An adjacency matrix of the same
        """
        time_ordered = np.array(time_ordered)
        if len(time_ordered.shape) == 1:
            n = int(np.sqrt(time_ordered.shape[0]))
            # It's a flattened vector, we must convert to a matrix first
            time_ordered = time_ordered.reshape((n, n))
        else:
            n = time_ordered.shape[0]
        order = np.where(time_ordered == 1)[1]
        adjacency = np.zeros((n, n))
        last = None
        for i, index in enumerate(order):
            if i == 0:
                pass
            else:
                adjacency[last, index] = 1
            last = index
        return adjacency

    def create_real(self, adjacency_matrix: np.ndarray):
        if np.sum(adjacency_matrix.shape) > self.n_qubits:
            raise ValueError('The adjacency matrix provided is too large')
        adj_vec = np.reshape(adjacency_matrix, self.n_qubits)
        for i, connection in enumerate(adj_vec):
            if connection:
                qml.RX((np.pi), wires=i)

    def generator(self, weights: List[Tuple[float]], **kwargs):
        for qb in range(self.n_qubits):
            qml.RX(weights[qb], wires=qb)

    def discriminator(self, weights: List[Tuple[float]], **kwargs):
        for qb in range(self.n_qubits):
            qml.RZ(weights[qb], wires=qb)
            qml.RX(weights[qb + self.n_qubits], wires=qb)
        qb_list = list(range(self.n_qubits))
        for i, (control, target) in enumerate(zip(qb_list[::-1], qb_list[::-1][1:])):
            qml.CNOT(wires=[control, target])
            #qGAN.iSWAP(weights[2 * self.n_qubits + i], wires=[control, target])


def create_qGAN(adjacency_matrix: np.ndarray, x_samples: List[np.ndarray]):
    n_qubits = adjacency_matrix.shape[0] ** 2
    qgan = qGAN(n_qubits)

    @qml.qnode(qgan.disc_dev, interface='tf')
    def real_disc_circuits(adjaceny_matrix: np.ndarray, disc_weights):
        qgan.create_real(adjacency_matrix=adjaceny_matrix)
        qgan.discriminator(disc_weights)
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(qgan.gen_dev, interface='tf')
    def gen_disc_circuits(gen_weights, disc_weights):
        qgan.generator(gen_weights)
        qgan.discriminator(disc_weights)
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(qgan.gen_dev, interface='tf')
    def generate_sample(gen_weights):
        qgan.generator(gen_weights)
        return [qml.expval(qml.PauliZ(x)) for x in range(n_qubits)]

    def real_true(sample_solution, disc_weights):
        disc_output = real_disc_circuits(sample_solution, disc_weights)
        return (disc_output + 1) / 2

    def fake_true(gen_weights, disc_weights):
        disc_output = gen_disc_circuits(gen_weights, disc_weights)
        return (disc_output + 1) / 2

    def disc_cost(sample_solution, gen_weights, disc_weight):
        cost = fake_true(gen_weights, disc_weight) - real_true(sample_solution, disc_weight)
        return cost

    def gen_cost(gen_weight, disc_weight):
        return - fake_true(gen_weight, disc_weight)

    def tsp_cost(sample_solution):
        return qGAN.tsp_cost(adjacency_matrix, sample_solution)

    def train_disc_step(x, gen_weights, disc_weights, optimiser):
        with tf.GradientTape() as tape:
            disc_loss = disc_cost(x, gen_weights, disc_weights)
        grads = tape.gradient(disc_loss, [disc_weights])
        optimiser.apply_gradients(zip(grads, [disc_weights]))
        return disc_loss

    def train_gen_step(x, gen_weights, disc_weights, optimiser):
        with tf.GradientTape() as tape:
            gen_loss = gen_cost(gen_weights, disc_weights)
        grads = tape.gradient(gen_loss, [gen_weights])
        optimiser.apply_gradients(zip(grads, [gen_weights]))
        return gen_loss

    def training(x_train):
        init_gen = np.random.normal(size=(n_qubits, ))
        init_disc = np.random.normal(size=((3 * n_qubits) - 1, ))
        gen_weights = tf.Variable(init_gen)
        disc_weights = tf.Variable(init_disc)

        optimiser = tf.optimizers.SGD(0.02)
        for e in range(15):
            for x in x_train:
                disc_loss = train_disc_step(x, gen_weights, disc_weights, optimiser)
                gen_loss = train_gen_step(x, gen_weights, disc_weights, optimiser)
            if not e % 5:
                print('Gen cost: {}\nDisc cost: {}'.format(gen_loss, disc_loss))

                print('Generated sample:\n{}'.format(np.round(generate_sample(gen_weights)).reshape(int(np.sqrt(n_qubits)),
                                                                                                   int(np.sqrt(n_qubits)))))

    training(x_samples)
