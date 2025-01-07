import networkx as nx
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class hexagonal_lattice:

    def __init__(self, m: int, n: int):
        """
        generates the connectivity lattice of the logical qubits
        has important methods such as the shortest first routing

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
        """
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
        start_x = copy.deepcopy(dct[(0, 0)][0])
        start_y = copy.deepcopy(dct[(0, 0)][1])
        for i, x_t in enumerate(range(max_x_tilde + 1)):
            if i == 0:
                y_range = range(1, max_y_tilde + 1)
            else:
                y_range = range(max_y_tilde + 1)
            for y_t in y_range:
                if y_t != 0:
                    start_x -= 1
                    start_y += 1
                if start_y % 2 == 0:
                    z = -start_x - start_y
                else:
                    z = -start_x - start_y + 1
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
                start_x = copy.deepcopy(dct[(x_t, 0)][0]) + 1
                start_y = copy.deepcopy(dct[(x_t, 0)][1]) + 1

        return dct

    def distance_triangular(self, pos_1: tuple, pos_2: tuple) -> int:
        """
        determines distance considering the triangular dual lattice

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
        lst = []
        assert len(mapped_1) == len(
            mapped_2
        ), "Something went wrong in the triangular mapping"
        for i in range(len(mapped_1)):
            lst.append(abs(mapped_1[i] - mapped_2[i]))
        dist = int(max(lst))
        return dist

    def plot_lattice(self) -> None:
        """
        plots the lattice G with networkx labels
        """

        pos = nx.get_node_attributes(self.G, "pos")

        plt.figure(figsize=(3.5, 3.5))
        nx.draw(self.G,
                pos, with_labels=True,
                node_color="lightgray",
                edge_color="lightblue")


class shortest_first_router(hexagonal_lattice):

    def __init__(
        self, m: int, n: int,
        terminal_pairs: list[tuple[tuple[int, int], tuple[int, int]]]
    ):
        """
        Start with graph $G$ and an empty solution.
        While $G$ contains any path connecting any demand pair,
        choose the shortest such path $P$, add $P$ to the solution,
        and delete all vertices of $P$ from $G$

        Args:
            m (int): The number of rows of hexagons in the lattice.
            n (int): The number of columns of hexagons in the lattice.
            terminal_pairs (list[tuple[tuple[int, int], tuple[int, int]]:
            pairs of vertices to be connected (networkx labeling)
        """
        super().__init__(m, n)
        self.terminal_pairs_orig = terminal_pairs.copy()
        self.terminal_pairs = terminal_pairs
        self.order_terminal_pairs()
        self.VDP_layers = self.find_all_VDP_layers()

        # ! generalize for multiple layers

    def split_layer_terminal_pairs(self):
        """
        split up the terminal pairs into layers which can be 
        compiled in parallel in principle because no qubits overlap
        """
        pass

    def order_terminal_pairs(self):
        """
        order the terminal pairs s.t. the pairs
        closest together are routed first
        adapts self.terminal_pairs in place
        """

        terminal_pair_dist = {}
        for t_p in self.terminal_pairs_orig:
            print("t_p", t_p)
            # paths must be found excluding other terminals
            G_temp = self.G.copy()
            terminal_pairs_flattened = [
                pair for sublist in self.terminal_pairs_orig
                for pair in sublist
            ]
            print("terminal_pairs_falttend", terminal_pairs_flattened)
            terminals_temp = [
                pair for pair in terminal_pairs_flattened
                if pair != t_p[0] and pair != t_p[1]
            ]
            terminals_temp = list(set(terminals_temp))
            print("terminals_temp", terminals_temp)
            G_temp.remove_nodes_from(terminals_temp)
            print("G_temp nodes", G_temp.nodes)
            try:
                path = nx.dijkstra_path(G_temp, t_p[0], t_p[1])
            except nx.NetworkXNoPath:
                raise ValueError("""
                Your choice of terminal pairs `locks` in at least one terminal.
                Reconsider your choice of terminal pairs.
                """)
            terminal_pair_dist.update({t_p: len(path)})

        sorted_terminal_pairs = sorted(
            terminal_pair_dist.keys(), key=lambda tp: terminal_pair_dist[tp]
        )
        self.terminal_pairs = sorted_terminal_pairs

    def find_max_VDP_set(
            self
    ) -> tuple[dict, list[tuple[tuple[int, int], tuple[int, int]]]]:
        """
        iteratively applies dijkstra and searches greedily the largest
        possible VDP set in this setting

        Returns:
            dict: path per terminal pair
            list[tuple[int,int]]: remaining terminal pairs which must be placed
                in a new layer
        """

        VDP_dict = {}
        terminal_pairs_remainder = []
        successful_terminals = []  # gather successful terminal pairs
        flag_problem = False
        G_temp = self.G.copy()
        # a dct which checks whether a qubit
        # was already used in the current layer
        dct_qubits = {}
        terminal_pairs_flattened = [
                pair for sublist in self.terminal_pairs_orig
                for pair in sublist
            ]
        for t in terminal_pairs_flattened:
            dct_qubits.update({t: False})
        dct_qubits_copy = dct_qubits.copy()
        print("dct_qubits", dct_qubits)
        for t_p in self.terminal_pairs:
            print("R: t_p", t_p)
            # path must be found excluding other terminals
            G_temp_temp = G_temp.copy()
            """terminal_pairs_temp = [
                pair for pair in self.terminal_pairs_orig if pair != t_p
            ]
            terminals_temp = [
                pair for sublist in terminal_pairs_temp for pair in sublist
            ]
            G_temp_temp.remove_nodes_from(terminals_temp)
            """
            if dct_qubits[t_p[0]] or dct_qubits[t_p[1]]:
                flag_problem = True
            else:
                print("R: terminal_pairs_falttend", terminal_pairs_flattened)
                terminals_temp = [
                    pair for pair in terminal_pairs_flattened
                    if pair != t_p[0] and pair != t_p[1]
                ]
                terminals_temp = list(set(terminals_temp))
                print("r: terminals_temp", terminals_temp)
                G_temp_temp.remove_nodes_from(terminals_temp)
                print("R:G_temp nodes", G_temp_temp.nodes)
                # find shortest path of t_p
                try:
                    path = nx.dijkstra_path(G_temp_temp, t_p[0], t_p[1])
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
            print("flag problem", flag_problem)
            # update already used qubits
            dct_qubits[t_p[0]] = True
            dct_qubits[t_p[1]] = True
            if flag_problem:
                terminal_pairs_remainder = [
                    s
                    for s in self.terminal_pairs
                    if s not in successful_terminals
                ]
                print("sucessfull terminals", successful_terminals)
                print("self.terminal_pairs", self.terminal_pairs)
                dct_qubits = dct_qubits_copy.copy()
            else:  # if no problem
                print("path", path)
                # remove nodes and edges from G
                # but only remove the path vertices, NOT the terminals
                # because the terminals might be used multiple times
                for node in path[1:-1]:
                    G_temp.remove_node(node)
                successful_terminals.append(t_p)
                VDP_dict.update({t_p: path})

        return VDP_dict, terminal_pairs_remainder

    def find_all_VDP_layers(self) -> list[dict]:
        """
        if find_max_VDP_set returns nonzero terminal_pairs_remainder
        it is required to run the algorithm as long s.t. we find all VDP
        sets even if they are in multiple layers

        Returns:
            list[dict]: list of layers with simultaneous paths (VDP per layer)
        """
        flag_continue = True
        VDP_layers = []
        while flag_continue:
            VDP_dict, terminal_pairs_remainder = self.find_max_VDP_set()
            print("VDP_dict", "remainder", VDP_dict, terminal_pairs_remainder)
            VDP_layers.append(VDP_dict)
            if len(terminal_pairs_remainder) == 0:
                flag_continue = False
                break
            else:
                self.terminal_pairs = terminal_pairs_remainder

        return VDP_layers

    def plot_lattice_paths(self, layer: int) -> None:
        """
        plots the graph and the corresponding VDP of a layer

        Args:
            layer (int): label of layer to plot
        """

        pos = nx.get_node_attributes(self.G, "pos")

        num_paths = len(self.VDP_layers[layer].keys())
        colormap = plt.cm.get_cmap("rainbow", num_paths)
        colors = [mcolors.to_hex(colormap(i)) for i in range(num_paths)]

        plt.figure(figsize=(3.5, 3.5))
        nx.draw(self.G, pos,
                with_labels=True,
                node_color="lightgray",
                edge_color="lightblue")

        for i, path in enumerate(self.VDP_layers[layer].values()):
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
