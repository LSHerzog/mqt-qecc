"""Hill Climbing with random restarts for Routing Layouts."""

from __future__ import annotations

import operator
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import rainbow
from tqdm import tqdm

from .lattice_router import HexagonalLattice, ShortestFirstRouter
from .misc import translate_layout_circuit


class HillClimbing:
    """Hill Climbing with random restarts for routing layouts."""
    def __init__(
            self, 
            max_restarts: int, 
            max_iterations: int, 
            circuit: list[tuple[int,int]], 
            layout_type: str, 
            m:int, 
            n:int, 
            metric: str
        ) -> None:
        """Initializes the Hill Climbing with Random Restarts algorithm.

        Args:
            max_restarts (int): Maximum number of random restarts.
            max_iterations (int): Maximum number of iterations per restart.
            circuit (list[tuple[int,int]]): list of qubits to connect (terminal pairs aka cnots) 
            layout_type (str): (row, sparse, pair)
            m (int): number of rows of hexagons in the lattice
            n (int): number of columns of hexagons in the lattices
            metric (str): "crossing", "routing", "distance"

        Raises:
            ValueError: _description_
        """
        self.m = m
        self.n = n
        self.max_restarts = max_restarts
        self.max_iterations = max_iterations
        assert layout_type in {"row", "sparse", "pair"}, "Unknown layout_type!"
        self.layout_type = layout_type
        lat = HexagonalLattice(m, n)
        if layout_type == "row":
            data_qubit_locs = lat.gen_layout_row()
        elif layout_type == "sparse":
            data_qubit_locs = lat.gen_layout_sparse()
        elif layout_type == "pair":
            data_qubit_locs = lat.gen_layout_pair()
        else:
            msg = "unkown layout type"
            raise ValueError(msg)
        self.data_qubit_locs = data_qubit_locs
        assert metric in {"crossing", "routing", "distance"}
        self.metric = metric
        self.circuit = circuit
        flattened_qubit_labels = [num for tup in self.circuit for num in tup]
        self.q = max(flattened_qubit_labels)+1
        assert len(list(set(flattened_qubit_labels))) == self.q, "The available qubits must allow a continuous labeling."
        assert len(data_qubit_locs)>=self.q, "The lattice must be able to host the number of qubits given in the circuit"

    def evaluate_solution(self, layout: dict) -> int:
        """Evaluates the layout=solution according to self.metric."""
        terminal_pairs = translate_layout_circuit(self.circuit, layout)
        router = ShortestFirstRouter(m = self.m, n = self.n, terminal_pairs = terminal_pairs)
        if self.metric == "crossing":
            cost = np.sum(router.count_crossings_per_layer())
        elif self.metric == "distance":
            distances = router.measure_terminal_pair_distances()
            cost = np.sum(distances)
        elif self.metric == "routing":
            vdp_layers = router.find_total_vdp_layers()
            cost = len(vdp_layers)
        return cost

    def gen_random_qubit_assignment(self) -> dict:
        """Yields a random qubit assignment given the `data_qubit_locs`."""
        layout = {}
        perm = list(range(self.q))
        random.shuffle(perm)
        for i,j in zip(perm, self.data_qubit_locs):
            layout.update({i: j})
        return layout
    
    def gen_neighborhood(self, layout: dict) -> list[dict]:
        """Creates the Neighborhood of a given layout by going through each terminal pair and swapping their positions.

        Args:
            layout (dict): qubit label assignment on the lattice. keys = qubit label, value = node label

        Returns:
            list[dict]: List of layouts constituting the neighborhood.
        """
        neighborhood = []
        for pair in self.circuit:
            layout_copy = layout.copy()
            #intermediate storage of the nodes 
            q_0_pos = layout_copy[pair[0]]
            q_1_pos = layout_copy[pair[1]]
            #swap
            layout_copy[pair[1]] = q_0_pos
            layout_copy[pair[0]] = q_1_pos
            neighborhood.append(layout_copy)

        return neighborhood
        

    def run(self) -> tuple[dict, int, int, dict]:
        """Executes the Hill Climbing algorithm with random restarts.

        Returns:
            best_solution: The best solution found.
            best_score: The score of the best solution.
        """
        best_solution = None
        best_rep = None
        best_score = float("inf")  # Use '-inf' for maximization, 'inf' for minimization
        score_history = {}

        for restart in tqdm(range(self.max_restarts), desc="Hill Climbing Restarts..."):
            current_solution = self.gen_random_qubit_assignment()
            current_score = self.evaluate_solution(current_solution)
            history_temp = {"scores" : [], "layout_init" : current_solution.copy()}
            for _ in range(self.max_iterations):
                neighbors = self.gen_neighborhood(current_solution)
                if not neighbors:
                    break  # No neighbors, end this restart

                # Find the best neighbor
                neighbor_scores = [(neighbor, self.evaluate_solution(neighbor)) for neighbor in neighbors]
                best_neighbor, best_neighbor_score = min(neighbor_scores, key=operator.itemgetter(1))  # Change to min for minimization

                # If no improvement, stop searching in this path
                if best_neighbor_score >= current_score:
                    break

                # Update current solution
                current_solution, current_score = best_neighbor, best_neighbor_score
                history_temp["scores"].append(current_score)

            history_temp.update({"layout_final" : current_solution.copy()})
            score_history.update({restart: history_temp})

            # Update global best solution if current is better
            if current_score < best_score:
                best_solution, best_score = current_solution, current_score
                best_rep = restart

        return best_solution, best_score, best_rep, score_history
    
    def plot_history(self, score_history: dict, filename: str = "./hc_history_plot.pdf", size: tuple[float,float] = (5,5)) -> None:
        """Plots the scores for each restart and iteration.

        Args:
            score_history (dict): Score history from HillClimber.run
            filename (str, optional): Path to store the plot. Defaults to "./hc_history_plot.pdf".
            size (tuple[float,float], optional): Size of the plot. Defaults to (3.5,3.5).
        """
        plt.figure(figsize=size)
        for rep, history in score_history.items():
            scores = history["scores"]
            plt.plot(range(len(scores)), scores, "x-", color = rainbow(rep/len(list(score_history.keys()))), label = f"Restart {rep}")
        plt.legend()
        plt.ylabel(f"{self.metric}")
        plt.xlabel("Hill Climbing Iteration")
        plt.title(f"$q=${self.q}, Layout-Type = {self.layout_type}, Num CNOTS = {len(self.circuit)}")
        plt.savefig(filename)