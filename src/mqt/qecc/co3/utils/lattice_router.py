"""Routing for Hexagonal Lattices."""

from __future__ import annotations

import copy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx


class HexagonalLattice:
    """Hexagonal Lattice with Distance Metric."""

    def __init__(self, m: int, n: int) -> None:
        """Generates the connectivity lattice of the logical qubits.

        Args:
            m (int): The number of rows of hexagons in the lattice.
            n (int): The number of columns of hexagons in the lattice.
        """
        self.m = m
        self.n = n
        self.G = nx.hexagonal_lattice_graph(
            m=m, n=n, periodic=False, with_positions=True, create_using=None
        )
        self.G_copy = copy.deepcopy(self.G)

    def map_hex_to_triangular(self) -> dict:
        """Maps positions of hex lattice to dual triangular lattice.

        maps the positions of the G lattice to a triangular lattice such that
        distances can be measured. follows along
        https://github.com/mhwombat/grid/wiki/Implementation:-Triangular-tiles

        Returns:
            dict: dictionary which maps between networkx labels key (x,y) to
                    triangular labels (x,y,z)
        """
        # generate the triangular lattice of the correct size
        max_x_tilde = max(
            el[0] for el in nx.get_node_attributes(self.G, "pos")
        )
        max_y_tilde = max(
            el[1] for el in nx.get_node_attributes(self.G, "pos")
        )
        full_max = max(max_x_tilde, max_y_tilde)

        # initialize large enough (double proof) such that only positive
        # integers can appear in triangular map
        dct = {
            (0, 0): (2 * full_max, 0, -2 * full_max)
        }
        start_x = copy.deepcopy(dct[0, 0][0])
        start_y = copy.deepcopy(dct[0, 0][1])
        for i, x_t in enumerate(range(max_x_tilde + 1)):
            y_range = range(1, max_y_tilde + 1) if i == 0 else range(max_y_tilde + 1)
            for y_t in y_range:
                if y_t != 0:
                    start_x -= 1
                    start_y += 1
                z = -start_x - start_y if start_y % 2 == 0 else -start_x - start_y + 1
                if self.G_copy.has_node((x_t, y_t)):
                    dct.update(
                        {
                            (x_t, y_t): (
                                copy.deepcopy(start_x),
                                copy.deepcopy(start_y),
                                copy.deepcopy(z),
                            )
                        }
                    )

            if x_t != max_x_tilde:
                start_x = copy.deepcopy(dct[x_t, 0][0]) + 1
                start_y = copy.deepcopy(dct[x_t, 0][1]) + 1

        return dct

    def distance_triangular(self, pos_1: tuple, pos_2: tuple) -> int:
        """Determines distance considering the triangular dual lattice.

        Args:
            pos_1 (tuple): position 1 on networkx graph G
            pos_2 (tuple): position 2 on networkx graph G

        Returns:
            int: distance between pos_1 and pos_2
        """
        assert (
            all(isinstance(x, int) for x in pos_1)
        ), "Each entry in pos_1 must be an integer!"

        assert (
            all(isinstance(x, int) for x in pos_2)
        ), "Each entry in pos_2 must be an integer!"

        dct = self.map_hex_to_triangular()
        mapped_1 = dct[pos_1]
        mapped_2 = dct[pos_2]
        lst = [abs(mapped_1[i] - mapped_2[i]) for i in range(len(mapped_1))]
        assert len(mapped_1) == len(
            mapped_2
        ), "Something went wrong in the triangular mapping"
        return int(max(lst))

    def plot_lattice(self) -> None:
        """Plots the lattice G with networkx labels."""
        pos = nx.get_node_attributes(self.G, "pos")

        plt.figure(figsize=(3.5, 3.5))
        nx.draw(self.G,
                pos, with_labels=True,
                node_color="lightgray",
                edge_color="lightblue")


class ShortestFirstRouter(HexagonalLattice):
    """Shortest First Routing for VDP on Hexagonal Lattice."""

    def __init__(
        self, m: int, n: int,
        terminal_pairs: list[tuple[tuple[int, int], tuple[int, int]]]
    ) -> None:
        """Routing for Hexagonal Lattice.
        
        Start with graph $G$ and an empty solution.
        While $G$ contains any path connecting any demand pair,
        choose the shortest such path $P$, add $P$ to the solution,
        and delete all vertices of $P$ from $G$

        Args:
            m (int): The number of rows of hexagons in the lattice.
            n (int): The number of columns of hexagons in the lattice.
            terminal_pairs (list[tuple[tuple[int, int], tuple[int, int]]): pairs of vertices to be connected (networkx labeling)
        """
        super().__init__(m, n)
        self.terminal_pairs_orig = terminal_pairs.copy()
        self.terminal_pairs = terminal_pairs
        self.layers_cnots = self.split_layer_terminal_pairs()
        self.layers_cnots_orig = self.layers_cnots.copy()
        self.vdp_layers = self.find_total_vdp_layers()

    def split_layer_terminal_pairs(self) -> list[list[tuple[int,int]]]:
        """Split Terminal Pairs into layers initially.

        split up the terminal pairs into layers which can be 
        compiled in parallel in principle because no qubits overlap
        """
        layers = []
        current_layer = [] 
        used_qubits = set()  

        for pair in self.terminal_pairs:
            if pair[0] in used_qubits or pair[1] in used_qubits:
                layers.append(current_layer)
                current_layer = [pair]
                used_qubits = set(pair)
            else:
                current_layer.append(pair)
                used_qubits.update(pair)

        if current_layer:
            layers.append(current_layer)

        return layers

    def order_terminal_pairs(self, layer: int) -> None:
        """Orders terminal pairs of a layer inplace.

        order the terminal pairs s.t. the pairs
        closest together are routed first
        adapts self.terminal_pairs in place
        """
        terminal_pair_dist = {}
        for t_p in self.layers_cnots_orig[layer]:#self.terminal_pairs_orig:
            # paths must be found excluding other terminals
            g_temp = self.G.copy()
            terminal_pairs_flattened = [
                pair for sublist in self.layers_cnots_orig[layer]#self.terminal_pairs_orig
                for pair in sublist
            ]
            terminals_temp = [
                pair for pair in terminal_pairs_flattened
                if pair != t_p[0] and pair != t_p[1]
            ]
            terminals_temp = list(set(terminals_temp))
            g_temp.remove_nodes_from(terminals_temp)
            try:
                path = nx.dijkstra_path(g_temp, t_p[0], t_p[1])
            except nx.NetworkXNoPath as exc:
                msg = (
                    "Your choice of terminal pairs locks in at least one terminal. "
                    "Reconsider your choice of terminal pairs."
                )
                raise ValueError(
                    msg
                ) from exc
            terminal_pair_dist.update({t_p: len(path)-1}) #-1 because we want to count only what is between the terminals
        sorted_terminal_pairs = sorted(
            terminal_pair_dist.keys(), key=lambda tp: terminal_pair_dist[tp]
        )
        #self.terminal_pairs = sorted_terminal_pairs
        self.layers_cnots_orig[layer] = sorted_terminal_pairs
        self.layers_cnots[layer] = sorted_terminal_pairs

    def find_max_vdp_set(
            self, layer: int
    ) -> tuple[dict, list[tuple[tuple[int, int], tuple[int, int]]]]:
        """Find largest VDP with shortest first.

        iteratively applies dijkstra and searches greedily the largest
        possible VDP set in this setting

        Returns:
            dict: path per terminal pair
            list[tuple[int,int]]: remaining terminal pairs which must be placed
                in a new layer
        """
        vdp_dict = {}
        terminal_pairs_remainder = []
        successful_terminals = []  # gather successful terminal pairs
        flag_problem = False
        g_temp = self.G.copy()
        # a dct which checks whether a qubit
        # was already used in the current layer
        dct_qubits = {}
        terminal_pairs_orig_current = self.layers_cnots_orig[layer].copy()
        terminal_pairs_current = self.layers_cnots[layer].copy()
        terminal_pairs_flattened = [
                pair for sublist in terminal_pairs_orig_current
                for pair in sublist
            ]
        for t in terminal_pairs_flattened:
            dct_qubits.update({t: False})
        dct_qubits_copy = dct_qubits.copy()
        for t_p in terminal_pairs_current:
            # path must be found excluding other terminals
            g_temp_temp = g_temp.copy()
            if dct_qubits[t_p[0]] or dct_qubits[t_p[1]]:
                flag_problem = True
            else:
                terminals_temp = [
                    pair for pair in terminal_pairs_flattened
                    if pair != t_p[0] and pair != t_p[1]
                ]
                terminals_temp = list(set(terminals_temp))
                g_temp_temp.remove_nodes_from(terminals_temp)
                # find shortest path of t_p
                try:
                    path = nx.dijkstra_path(g_temp_temp, t_p[0], t_p[1])
                except nx.NetworkXNoPath:
                    # if no path could be found: stop and return remaining,
                    # unallocated terminal pairs as well
                    """
                    terminal_pairs_remainder = [
                        s
                        for s in self.terminal_pairs
                        if s not in successful_terminals
                    ]
                    """
                    flag_problem = True
                    # break
            # update already used qubits
            dct_qubits[t_p[0]] = True
            dct_qubits[t_p[1]] = True
            if flag_problem:
                terminal_pairs_remainder = [
                    s
                    for s in terminal_pairs_current
                    if s not in successful_terminals
                ]
                dct_qubits = dct_qubits_copy.copy()
            else:  # if no problem
                # remove nodes and edges from G
                # but only remove the path vertices, NOT the terminals
                # because the terminals might be used multiple times
                for node in path[1:-1]:
                    g_temp.remove_node(node)
                successful_terminals.append(t_p)
                vdp_dict.update({t_p: path})

        return vdp_dict, terminal_pairs_remainder

    def find_all_vdp_layers(self, layer: int) -> list[dict]:
        """Find VDP layers within a given initial layer.

        if find_max_VDP_set returns nonzero terminal_pairs_remainder
        it is required to run the algorithm as long s.t. we find all VDP
        sets even if they are in multiple layers

        Returns:
            list[dict]: list of layers with simultaneous paths (VDP per layer)
        """
        flag_continue = True
        vdp_layers = []
        while flag_continue:
            vdp_dict, terminal_pairs_remainder = self.find_max_vdp_set(layer)
            vdp_layers.append(vdp_dict)
            if len(terminal_pairs_remainder) == 0:
                flag_continue = False
                break
            self.layers_cnots[layer] = terminal_pairs_remainder

        return vdp_layers
    
    def find_total_vdp_layers(self) -> list[dict]:
        """Find all routes for all initial and secondary layers.

        finds total VDP layers, i.e. more than `all` meaning that 
        it also respects the initial layer structure of the cnots
        """
        vdp_layers = []
        for layer in range(len(self.layers_cnots_orig)):
            self.order_terminal_pairs(layer)
            vdp_layers_temp = self.find_all_vdp_layers(layer)
            vdp_layers += vdp_layers_temp
        return vdp_layers

    def plot_lattice_paths(self, layer: int) -> None:
        """Plots the graph and the corresponding VDP of a layer.

        Args:
            layer (int): label of layer to plot
        """
        pos = nx.get_node_attributes(self.G, "pos")

        num_paths = len(self.vdp_layers[layer].keys())
        colormap = plt.cm.get_cmap("rainbow", num_paths)
        colors = [mcolors.to_hex(colormap(i)) for i in range(num_paths)]

        plt.figure(figsize=(3.5, 3.5))
        nx.draw(self.G, pos,
                with_labels=True,
                node_color="lightgray",
                edge_color="lightblue")

        for i, path in enumerate(self.vdp_layers[layer].values()):
            if path:
                path_edges = [
                    (path[j], path[j + 1]) for j in range(len(path) - 1)
                ]
                nx.draw_networkx_edges(
                    self.G,
                    pos,
                    edgelist=path_edges,
                    width=2,
                    edge_color=colors[i]
                )
                nx.draw_networkx_nodes(
                    self.G,
                    pos,
                    nodelist=path,
                    node_color=colors[i],
                    label=f"Path {i + 1}"
                )

        plt.legend()
        plt.show()
