"""Generates Stabilizers for n-snakes."""

from __future__ import annotations

import itertools
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


class SnakeBuilderSC:
    """Constructs a n-snake of distance d with a surface code in snake shape on the square lattice substate (leading to brickwall routing graph aka hex graph)."""
    def __init__(
            self,
            g: nx.Graph,
            positions_rough: list[list[list[tuple[int,int]]]],
            positions_smooth: list[list[list[tuple[int,int]]]],
            d: int
    ) -> None:
        """Initializes a SC n-snake.

        Args:
            g (nx.Graph): quadratic graph
            positions_rough (list[list[list[tuple[int,int]]]]): Node positions constituting the rough boundaries
            positions_smooth (list[list[list[tuple[int,int]]]]): Node positions constituting the smooth boundaries
            d (int): distance
        """
        self.g = g
        self.positions_rough = positions_rough
        self.positions_smooth = positions_smooth
        self.d = d
        assert len(positions_smooth) == 2, "To encode 1 logical qubit there must be 2 smooth boundaries."
        assert len(positions_rough) == 2, "To encode 1 logical qubit there must be 2 rough boundaries."
        dist = min([len(el) for el in positions_rough] + [len(el) for el in positions_smooth]) - 1 #-1 because we count edges not nodes
        assert dist == d, f"Distance d={dist} does not coincide with the geometry of the rough and smooth positions."




    def fill_snake(self) -> list[list[tuple[int,int]]]:
        """Adds the inner nodes given by the boundary `positions`. Sweeps through rows of lattice.

        Returns:
            list[list[tuple[int,int]]]: Positions of ALL nodes in the snakes.
        """
        positions_smooth_flattened = [pos for sublist in self.positions_smooth for pos in sublist]
        positions_rough_flattened = [pos for sublist in self.positions_rough for pos in sublist]
        positions = list(set(positions_smooth_flattened + positions_rough_flattened))#remove duplicate elements because corners appear twice.
        inside_nodes = set()

        x_values = {x for x, _ in self.g.nodes()}
        y_values = {y for _, y in self.g.nodes()}

        #sweep row by row of the lattice
        for y in sorted(y_values):
            inside = False
            temp_nodes = []
            for x in sorted(x_values):
                node = (x,y)
                if node in positions:
                    if inside:
                        inside_nodes.update(temp_nodes)
                    inside = not inside
                    temp_nodes = []
                elif inside:
                    temp_nodes.append(node)

        self.inner_nodes = list(inside_nodes)
        self.boundary_nodes = list(positions)
        
        return positions + list(inside_nodes)
    
    @staticmethod
    def neighbors_ver_hor(node1: tuple[int,int], node2: tuple[int,int]) -> bool:
        """Checks whether two nodes are neighbors.

        Vertical and horizontal neighbors. Not based on initial graph structure because no diagonal edges present.

        Args:
            node1 (tuple[int,int]): node1
            node2 (tuple[int,int]): node2

        Returns:
            bool: whether both are neighbors.
        """
        if abs(node1[0] - node2[0])==1 and node1[1] == node2[1]: #vertical neighbors
            return True
        if abs(node1[1] - node2[1])==1 and node1[0] == node2[0]: #horizontal neighbors
            return True
        return False

    @staticmethod
    def neighbors_diag(node1: tuple[int,int], node2: tuple[int,int]) -> bool:
        """Checks whether two nodes are neighbors.

        Diagonal. Not based on initial graph structure because no diagonal edges present.

        Args:
            node1 (tuple[int,int]): node1
            node2 (tuple[int,int]): node2

        Returns:
            bool: whether both are neighbors.
        """
        return bool(abs(node1[1] - node2[1]) == 1 and abs(node1[0] - node2[0]) == 1) #diagonal neighbors or no neighbors at all


    def collect_qubit_positions(self) -> list[tuple[tuple[int,int], tuple[int,int]]]:
        """Collect the edges defining the qubits we need, depending on rough/smooth edges.
        
        If qubits are placed on the diagonal, add the corresponding edge to self.g 

        Returns:
            list[tuple[tuple[int,int], tuple[int,int]]]: List of edges where qubits are placed.
        """
        qubit_edges = []
        
        #inner qubits
        for pair in itertools.combinations(self.inner_nodes, 2): #could be done more systematically
            neighbor_bool = self.neighbors_ver_hor(pair[0], pair[1])
            if neighbor_bool:
                qubit_edges.append(pair)
                if pair not in self.g.edges():
                    self.g.add_edge(*pair)

        #qubits on edges between inner and boundary vertices
        for pair in itertools.product(self.boundary_nodes, self.inner_nodes):
            neighbor_bool = self.neighbors_ver_hor(pair[0], pair[1])
            if neighbor_bool:
                qubit_edges.append(pair)
                if pair not in self.g.edges():
                    self.g.add_edge(*pair)

        #smooth boundary qubits (no qubits added between vertices on the rough boundary)
        for smooth_boundary in self.positions_smooth:
            for pair in itertools.combinations(smooth_boundary, 2):
                neighbor_bool = self.neighbors_ver_hor(pair[0], pair[1]) or self.neighbors_diag(pair[0], pair[1])
                if neighbor_bool:
                    qubit_edges.append(pair)
                    if pair not in self.g.edges():
                        self.g.add_edge(*pair)

        
        #also check neighborhood between the boundaries (you might miss some qubit placements otherwise)
        for boundary_pair in itertools.product(self.positions_rough, self.positions_smooth):
            #make sure that the positions_rough list and the positions_smooth are disjoint (remove overlapping corner elements from one of the lists)
            set1 = set(boundary_pair[0])
            set2 = set(boundary_pair[1])
            common_corners = set1 & set2
            boundary1 = list(set1-common_corners)
            boundary2 = list(set2-common_corners)
            for pair in itertools.product(boundary1, boundary2):
                neighbor_bool = self.neighbors_ver_hor(pair[0], pair[1]) #or self.neighbors_diag(pair[0], pair[1])
                #need an additional constraint which checks that the pair does not connect two nodes on the rough boundary
                for rough_boundary in self.positions_rough:
                    pairs_rough = list(itertools.combinations(rough_boundary,2))
                    #print("pairs_rough", pairs_rough)
                    if (pair[0], pair[1]) in pairs_rough or (pair[1], pair[0]) in pairs_rough:
                        rough_bool = True
                    else:
                        rough_bool = False
                if neighbor_bool and not rough_bool:
                    qubit_edges.append(pair)
                    if pair not in self.g.edges():
                        self.g.add_edge(*pair)
        

        self.qubit_edges = qubit_edges
        return qubit_edges

    def gen_stars(self) -> list[list[tuple[tuple[int,int], tuple[int,int]]]]:
        """Generates star operators.

        Returns:
            list[list[tuple[tuple[int,int], tuple[int,int]]]]: List of lists of edges, where each list determines a star operator.
        """
        stars = []
        nodes = self.boundary_nodes + self.inner_nodes #all nodes
        for node in nodes:
            #collect all qubits which are connected to this node
            temp_qubits = [edge for edge in self.qubit_edges if edge[0] == node or edge[1] == node]
            if len(temp_qubits)>2:
                stars.append(temp_qubits)
        self.stars = stars
        return stars
    
    def gen_plaquettes(self) -> list[list[tuple[tuple[int,int], tuple[int,int]]]]:
        """Generates Plaquette Operators.

        Returns:
            list[list[tuple[tuple[int,int], tuple[int,int]]]]: _description_
        """
        plaquettes = []
        nodes = self.boundary_nodes + self.inner_nodes #all nodes
        #since we check the nodes for being in the upper left corner, you can loose some plaquettes. hence add more nodes to minimum x and y, even though there will be useless checks
        min_x = min(t[0] for t in nodes)
        min_y = min(t[1] for t in nodes)
        collected = []
        i, j = 0, 0

        while True:
            point = (min_x + i, min_y + j)
            if point in set(nodes):
                j = 0
                i += 1
                continue
            collected.append(point)

            j += 1
            if j > max(t[1] for t in nodes): 
                j = 0
                i += 1
            if i > max(t[0] for t in nodes): 
                break
        nodes += collected

        #find all squares in the snake
        #for each square, check where qubits are and add to plaquettes (number of qubits can be smaller than 4)
        for node in nodes:
            #create square of nodes with node in upper left corner. 
            square = [node, (node[0], node[1]+1), (node[0]+1, node[1]+1), (node[0]+1, node[1])] #cyclically aligned
            #edges_square = list(zip(square, square[1:] + [square[0]]))
            edges_square = list(itertools.combinations(square, 2))#also diagonals
            #check which qubits (edges) are contained
            #order edges square and qubit edges
            #edges_square = sorted(edges_square)
            #qubit_edges = sorted(self.qubit_edges)
            plaquette = [edge for edge in edges_square if (edge[0], edge[1]) in self.qubit_edges or (edge[1], edge[0]) in self.qubit_edges]
            #add to plaquettes if more than 1 qubit contained
            if len(plaquette)>1:
                plaquettes.append(plaquette)
        self.plaquettes = plaquettes
        return plaquettes

    def create_stabs(self) -> tuple[list[list[tuple[tuple[int,int], tuple[int,int]]]], list[list[tuple[tuple[int,int], tuple[int,int]]]]]:
        """Summarizes all Methods here."""
        _ = self.fill_snake()
        _ = self.collect_qubit_positions()
        _ = self.gen_stars()
        _ = self.gen_plaquettes()
        
        q = len(self.qubit_edges)
        lenstars = len(self.stars)
        lenplaq = len(self.plaquettes)
        assert q - lenstars - lenplaq == 1, f"Your stabilizers are wrong. They should create 1 logical qubit but they yield {q - lenstars - lenplaq} instead."

        return self.plaquettes, self.stars

        
        #add check whether ther are q-num stabs = 1
    def gen_checks(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return checks and translation dict."""
        trans_dict = {edge: i for i, edge in enumerate(self.qubit_edges)} #both edge orderings should be included to make sure that we get no key errors
        trans_dict2 = {(edge[1], edge[0]): i for i, edge in enumerate(self.qubit_edges)}
        trans_dict |= trans_dict2
        q = len(self.qubit_edges)
        hz = np.zeros((len(self.plaquettes), q), dtype=int)
        for row, plaquette in enumerate(self.plaquettes):
            translated_plaquette = [int(trans_dict[edge]) for edge in plaquette]
            for el in translated_plaquette:
                hz[row, el] = 1
        hx = np.zeros((len(self.stars), q), dtype=int)
        for row, star in enumerate(self.stars):
            translated_star = [int(trans_dict[edge]) for edge in star]
            for el in translated_star:
                hx[row, el] = 1
        self.trans_dict = trans_dict
        return hx, hz, trans_dict

    def plot_stabs(self, opz: list | None = None, opx: list | None = None) -> None:
        """Plots plaquettes and star operators as well as the snake itself.
        
        opz and opx are the logical operators retrieved via mqt.qecc.CSSCode which are already translated as edges on the graph.
        """
        pos = {node: (node[0], -node[1]) for node in self.g.nodes()}  # Adjust for proper display

        midpoints = [((x1 + x2) / 2, -(y1 + y2) / 2) for (x1, y1), (x2, y2) in self.qubit_edges]

        plt.figure(figsize=(8,8))
        nx.draw(self.g, pos, with_labels=True, node_size=100, edge_color="lightgray", font_size = 8)

        for star in self.stars:
            all_nodes = [node for edge in star for node in edge]
            node_counts = Counter(all_nodes)
            central_node = max(node_counts, key=node_counts.get)
            for u,v in star:
                if u == central_node:
                    start, end = u, v
                elif v == central_node:
                    start, end = v, u
                else:
                    continue 

                midpoint = ((pos[start][0] + pos[end][0]) / 2, (pos[start][1] + pos[end][1]) / 2)
                
                # Plot the half-edge from the central node to the midpoint
                plt.plot([pos[start][0], midpoint[0]], [pos[start][1], midpoint[1]], color="orange", linewidth=3, label="X Star")

        ax = plt.gca()
        for plaquette in self.plaquettes:
            square = {node for edge in plaquette for node in edge}
            square_pos = [pos[node] for node in square] 
            square_pos = convex_hull(square_pos)
            #shrink the polygon a little
            square_pos = np.array(square_pos)
            centroid = square_pos.mean(axis=0)
            factor = 0.6
            square_pos = centroid + factor * (square_pos - centroid)
            polygon = Polygon(square_pos, closed=True, color="green", alpha=0.3, label="Z Face") 
            ax.add_patch(polygon)

        nodes = self.boundary_nodes + self.inner_nodes #all nodes
        x_mid, y_mid = zip(*midpoints)  # Extract x and y coordinates
        plt.scatter(x_mid, y_mid, color="red", s=20, zorder=3)  # Small blue dots
        nx.draw_networkx_nodes(self.g, pos, nodelist=nodes, node_color="blue", node_size=100)

        #integer labels
        for original_label, new_label in self.trans_dict.items():
            if original_label[0] in pos and original_label[1] in pos:  # Ensure both nodes exist
                x1, y1 = pos[original_label[0]]
                x2, y2 = pos[original_label[1]]

                # Compute the midpoint
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                # Plot the text at the midpoint
                plt.text(
                    mid_x, mid_y, str(new_label), fontsize=8,
                    color="green", ha="center", va="center"
                )

        if opx is not None:
            nx.draw_networkx_edges(self.g, pos, edgelist=opx, edge_color="teal", width=5, alpha= 0.7, label = "$X_L$")
        if opz is not None:
            nx.draw_networkx_edges(self.g, pos, edgelist=opz, edge_color="blueviolet", width=5, alpha = 0.7, label = "$Z_L$")


        #no duplicates in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_legend = dict.fromkeys(labels, None)  # Removes duplicates while keeping order
        unique_handles = [handles[labels.index(label)] for label in unique_legend]
        plt.legend(unique_handles, unique_legend.keys())
        plt.show()



class SnakeBuilderSTDW:
    """Constructs a n-snake of distance d color codes with semi transparent domain wall."""
    def __init__(
            self,
            g: nx.Graph,
            positions: list[list[tuple[int,int]]],
            d: int 
    ) -> None:
        """Initializes a n snake with ancillas on the interface and semi transparent domain wall.

        Args:
            g (nx.Graph): Hexagonal graph on which the data qubits are placed
            positions (list[tuple[int,int]]): vertex positions on the nx graph (tuples) for each triangular color code patch  
                The order of the triangles is important, as subsequent triangle lists shoudl belong to neighboring triangles.
            d (int): distance of the triangular color code patches
        """
        self.g = g
        self.positions = positions
        self.d = d

        #determine global labeling for all vertices in the n-snake
        all_pos = []
        for tile in self.positions:
            all_pos += tile
        labels = {}
        for i, pos in enumerate(all_pos):
            labels.update({pos : i})
        self.labels = labels

        #assertions for q(d) per triangle and p(d)=(q(d)-1)/2 per triangle (number of patches) 
        t = (d-1)/2
        q = int(3*t**2 + 3*t + 1)
        for i, triangle in enumerate(self.positions):
            assert len(triangle) == q, f"Your set of vertices for triangle {i} does not fit the expected number of qubits for distance d={d}."
        p = int((q-1)/2)
        self.q = q
        self.p = p
        self.n = len(positions)
        self.q_tilde = self.n * self.q + (self.n - 1) * (self.d - 1)
        self.trans_dict = None

    def find_triangle_edges_corners(self, n_triangle: int) -> list[list[tuple[int,int]]]:
        """Searches for the graph labels of the vertices in the 3 edges of the given triangle.

        Args:
            n_triangle (int): label of triangle in self.positions

        Returns:
            list[list[tuple[int,int]]]: list of three edges
        """
        triangle = self.positions[n_triangle]
        lst_boundary = [] #gather all vertices with a single neighbor outside the set (i.e. on edge)
        lst_corner = [] # gather all 3 vertices with 2 neighbors outside the set (i.e. corners of triangle)
        for vertex in triangle:
            neighbors_temp = list(self.g.neighbors(vertex))
            outside_neighbors = [v for v in neighbors_temp if v not in triangle]
            if len(outside_neighbors) == 2:
                lst_corner.append(vertex)
            elif len(outside_neighbors) == 1:
                lst_boundary.append(vertex)
            elif len(outside_neighbors) == 0:
                continue
            elif len(outside_neighbors) == 3:
                msg = f"There is an isolated qubit in your input triangle {n_triangle}."
                raise ValueError(msg)
        assert len(lst_corner) == 3, f"Something weird happened. lst_corner has {len(lst_corner)} elements instead of 3."
        assert len(lst_boundary) == (self.d-2)*3, f"Something weird happened. lst_boundary has {len(lst_boundary)} elements instead of {(self.d-2)*3}."

        return [lst_corner, lst_boundary]
    
    def find_interface_ancillas(self, triangle_0: int, triangle_1: int) -> list[tuple]:
        """Finds ancilla vertices on the interface between triangle_0 and triangle_1.

        Args:
            triangle_0 (int): label of first triangle in self.positions[triangle_0]
            triangle_1 (int): label of second triangle in self.positions[triangle_1] -> must be adjacent to the first!

        Returns:
            list[tuple] pair of ancilla locations which are in the interface.
        """
        [lst_corner_0, lst_boundary_0] = self.find_triangle_edges_corners(triangle_0)
        [lst_corner_1, lst_boundary_1] = self.find_triangle_edges_corners(triangle_1)

        #find pairs of elements which are next nearest neighbors.
        next_nearest_neighbors = []
        for el0 in lst_corner_0:
            for el1 in lst_corner_1:
                path = nx.dijkstra_path(self.g, el0, el1)
                if len(path)-1 == 2:
                    next_nearest_neighbors.append({"el0": el0, "el1": el1, "path": path})
        for el0 in lst_boundary_0:
            for el1 in lst_boundary_1:
                path = nx.dijkstra_path(self.g, el0, el1)
                if len(path)-1 == 2:
                    next_nearest_neighbors.append({"el0": el0, "el1": el1, "path": path})

        #filter ancillas
        ancillas = [el["path"][1] for el in next_nearest_neighbors]
        
        #only use those ancillas which do indeed have a nearest neighbor in the ancilla set (single ancillas not wanted)
        ancilla_pairs = []
        for node in ancillas:
            neighbors = set(self.g.neighbors(node))
            valid_neighbors = neighbors.intersection(ancillas)
            if valid_neighbors:
                # Ensure unique pairs, avoid (n1, n2) and (n2, n1)
                ancilla_pairs.extend((node, neighbor) for neighbor in valid_neighbors if node < neighbor)

        return ancilla_pairs
    
    def hex_plaquettes(self) -> list:
        """Find all hexagonal plaquettes on original g.

        Returns:
            list: all possible hexagonal plaquettes as vertices on g.
        """
        cycles = list(nx.simple_cycles(self.g, length_bound = 6))
        return [set(cycle) for cycle in cycles if len(cycle) == 6]

    def find_stabilizers(self) -> tuple[list,list]:
        """Find stabilizers on self.positions.
        
        Returns:
            tuple[list, list]: Two lists, first the Z stabilizrs, and second the X stabilizers. There are more Z than X stabilizers
            because we assume a Z merge by default. but can be interchanged of course for a X merge.
        """ 
        total_nodes = [] #find all relevant nodes first
        z_plaquettes = []
        
        for triangle in self.positions: #all nodes in the triangles
            total_nodes += triangle

        for i in range(len(self.positions) - 1): # everything in the interface
            ancilla_pairs = self.find_interface_ancillas(i, i+1)
            z_plaquettes += ancilla_pairs#pair stabs
            ancillas_flattened = [item for sublist in ancilla_pairs for item in sublist]
            total_nodes += ancillas_flattened
            
        #structure the nodes as intersection to the underlying hexagonal plaquettes (automatically other shape in interface)
        hexagonal_plaquettes = self.hex_plaquettes()
        for plaquette in hexagonal_plaquettes:
            overlap = set(plaquette) & set(total_nodes)
            if len(overlap) >= 3:  # Ensure a meaningful plaquette (full or partial)
                z_plaquettes.append(overlap)
        
        #filter out interface only plaquettes to distinguish x_plaquettes and z_plaquettes
        x_plaquettes = []
        for plaquette in z_plaquettes:
            if len(plaquette) == 6: #pairs, weight-3, weight-5 in the interface NOT wanted for X stabs
                #also remove the hex plaquettes within the interface (touching vertices of two triangles)
                bools = []
                for i in range(len(self.positions) - 1):
                    if set(plaquette) & set(self.positions[i]) and set(plaquette) & set(self.positions[i+1]):
                        bools.append(False)
                    else:
                        bools.append(True)
                if all(bools):
                    x_plaquettes.append(plaquette) #only if above NOT fulfilled
            elif len(plaquette) == 4:
                x_plaquettes.append(plaquette)

        self.total_nodes = total_nodes
        # build in assertion regarding number of each stabilizers, i have equations to check whether the number is right.
        assert len(x_plaquettes) == self.n * self.p, "Your number of final x_plaquettes is wrong, maybe weird input?"
        assert len(z_plaquettes) == self.n * self.p + self.d*(self.n - 1), "Your number of final z_plaquettes is wrong, maybe weird input?"
        
        return z_plaquettes, x_plaquettes
    
    def find_separate_stabilizers(self, n_triangle: int) -> list:
        """Generates the stabilizers of a plain triangular color code. Not including the interface.

        Args:
            n_triangle (int): index of triangular patch of interest.

        Returns:
            list: stabilizers of the code (both x and z stabilizers because self-dual)
        """
        plaquettes = []
        hexagonal_plaquettes = self.hex_plaquettes()
        for plaquette in hexagonal_plaquettes:
            overlap = set(plaquette) & set(self.positions[n_triangle])
            if len(overlap) >= 3:  # Ensure a meaningful plaquette (full or partial)
                plaquettes.append(overlap)

        assert len(plaquettes) == self.p, "Your number of final triangular color code plaquettes is wrong."
        return plaquettes    

    def integer_labeling(self) -> None:
        """Finds a random integer labeling. Only works after having run find_stabilizers."""
        trans_dict = {}
        for i, node in enumerate(self.total_nodes):
            trans_dict.update({node: i})
        self.trans_dict = trans_dict
    
    def plot_stabilizers(self, plaquettes: list, size: tuple[int, int] = (7, 7)) -> None:
        """Plots the stabilizers, either z_plaquettes or x_plaquettes."""
        pos = nx.get_node_attributes(self.g, "pos")
        plt.figure(figsize=size)
        nx.draw(self.g, pos, with_labels=True, font_size=8, node_color="lightgray", edge_color="lightblue")

        #integer labels
        if self.trans_dict is None:
            self.integer_labeling()
        for original_label, new_label in self.trans_dict.items():
            if original_label in pos:  # Ensure the node exists in the graph
                x, y = pos[original_label]
                plt.text(
                    x, y + 0.2, str(new_label), fontsize=8,
                    color="blue", ha="center", va="center"
                )

        #outline of the triangles
        for i in range(len(self.positions)):
            [lst_corner, _] = self.find_triangle_edges_corners(i)
            #plot three connection lines
            triangle_pos = [pos[node] for node in lst_corner]
            x_coords, y_coords = zip(*triangle_pos)
            plt.plot(
                (*x_coords, x_coords[0]),  # Close the triangle
                (*y_coords, y_coords[0]),  
                linewidth=3, color="black", alpha=0.5  # Thick and semi-transparent
            )


        colors = plt.cm.rainbow(np.linspace(0, 1, len(plaquettes)))
        for idx, face in enumerate(plaquettes):
            # Get the positions for the vertices in the face
            face_positions = [pos[node] for node in face]
            
            if len(face_positions) == 2:
                v1, v2 = face_positions[0], face_positions[1]
                line = Line2D([v1[0], v2[0]], [v1[1], v2[1]], color=colors[idx], lw=4)  # 'lw' is line width
                plt.gca().add_line(line)
            else:
                face_positions = convex_hull(face_positions)
                polygon = Polygon(face_positions, closed=True, edgecolor="blue", facecolor=colors[idx], alpha=0.6)
                plt.gca().add_patch(polygon)

        plt.show()
        # !todo store also the pdf of the figure in a given path.

    def gen_check_matrix(self, plaquettes: list) -> list:
        """Takes plaquettes and translates with self.integer_labeling."""
        self.integer_labeling()
        h = np.zeros((len(plaquettes), self.q_tilde), dtype=int)
        for row, plaquette in enumerate(plaquettes):
            translated_plaquette = [int(self.trans_dict[node]) for node in plaquette]
            for el in translated_plaquette:
                h[row, el] = 1
        return h


class SnakeBuilder:
    """Constructs a snake with n Steane patches on specified vertices in G. Without ancillas in the interface."""
    def __init__(
            self,
            g: nx.Graph,
            positions: list[dict]
    ) -> None:
        """Initializes n snake.

        Args:
            g (nx.Graph): Hexagonal graph on which the data qubits are placed
            positions (list[dictionary]): The positions must have key = networkx label, value = 0,..6. The labeling from 0-6 for each
                steane patch (each patch has one dictionary). This follows a strict convention, the order of the overall list is important
                since consecutive dictionaries must have patches neighboring on the lattice. Each patch has three edges: (0,2,1), (3,5,1), (0,4,3).
                Your current patch has to be connected with the next patch via a (3,5,1) -  (0,2,1) or (0,4,3) - (0,2,1) connection. This means
                you always have to `dock` your new patch with its (0,2,1) patch to the previous patch. note that (0,2,1) -(0,2,1) connections are NOT allowed.
                the ordering of 0-6 per patch must follow the convention such that self.standard_steane is consistent.
                     3
                   / | \
                  5--6--4  
                 /   |   \
                2----1----0
        """
        self.g = g
        self.positions = positions
        self.n = len(positions)
        for tile in self.positions:
            assert sorted(tile.values()) == list(range(7)), "Your 0-6 labeling of each Steane Tile is wrong!"
        
        #determine global labeling for all vertices in the n-snake
        all_pos = []
        for tile in self.positions:
            all_pos += list(tile.keys())
        labels = {}
        for i, pos in enumerate(all_pos):
            labels.update({pos : i})
        self.labels = labels

    @staticmethod
    def compatible_x_stabs() -> list[dict]:
        """Returns the allowed Weight 8 X Stabilizers crossing Steane patches."""
        return [
            {"i": [0,2,4,6], "i+1": [0,2,4,6]},
            {"i": [3,4,5,6], "i+1": [1,2,5,6]},
            {"i": [3,4,5,6], "i+1": [0,2,4,6]},
            {"i": [1,2,5,6], "i+1": [1,2,5,6]}
        ]
    
    @staticmethod
    def compatible_z_stabs_weight_two() -> list[dict]:
        """Returns allowed weight-2 z stabilizer connections."""
        return [
            {"i": 1, "i+1": 1},
            {"i": 0, "i+1": 0},
            {"i": 3, "i+1": 1},
            {"i": 1, "i+1": 3},
            {"i": 3, "i+1": 0},
            {"i": 0, "i+1": 3},
            {"i": 1, "i+1": 0},
            {"i": 0, "i+1": 1},
        ]
    
    @staticmethod
    def compatible_z_stabs_weight_four() -> list[dict]:
        """Returns allowed weight-4 z stabilizers connections."""
        return [
            {"i": [0,4], "i+1": [0,2]},
            {"i": [3,4], "i+1": [1,2]},
            {"i": [1,5], "i+1": [1,2]},
            {"i": [3,5], "i+1": [0,2]}
        ]
    
    @staticmethod
    def standard_steane() -> list[list[int]]:
        """Returns the standard separate steane stabilizer plaquettes."""
        return [
            [0,2,4,6],
            [1,2,5,6],
            [3,4,5,6]
        ]

    def neighboring_vertex(self, vertex_0: tuple, vertex_1: tuple) -> bool:
        """Checks whether two vertices are adjacent."""
        neighbor = False
        if (vertex_0, vertex_1) in self.g.edges() or (vertex_1, vertex_0) in self.g.edges():
            neighbor = True
        return neighbor
    
    def check_interface(self, i: int) -> dict:
        """Checks which edge of the ith steane tile is connected to the next (i+1) 0,2,1 edge."""
        next_edge = [key for key, value in self.positions[i+1].items() if value in {0,1,2}]
        #find adjacent edge of ith steane to `next_edge`'s 0,1,2 edge
        adjacent_vertices = set() #whole set of adjacent vertices to vertices in `next_edge`
        for vertex in next_edge:
            adjacent_vertices.update(self.g.neighbors(vertex))
        #check which nodes from ith steane are in adjacent_vertices
        adjacent_edge = {
            vertex: self.positions[i][vertex]
            for vertex in adjacent_vertices
            if vertex in self.positions[i]
        }
        assert len(adjacent_edge) == 3, "Something with the input steane tiles must be wrong (incorrect number of adjacent vertices. should be 3)"
        return adjacent_edge
    
    def check_paired_neighbor(self, pos_i_new: dict, pos_i1_new: dict) -> bool:
        """Checks whether we can find a weight-8 x plaquette which actually connects neighbored plaqeuttes between i and i+1."""
        neighboring_pairs = []  
        neighboring_two = False
        #neighboring_two is a bool which determines whether the weight 8 stab would connect neighboring patches
        for key_i in pos_i_new:
            for key_i1 in pos_i1_new:
                # Check if the pair of keys are neighbors using the neighboring_vertex function
                if self.neighboring_vertex(key_i, key_i1):
                    neighboring_pairs.append((key_i, key_i1))  # Store the pair
                    # If we already found 2 pairs, return true
                    if len(neighboring_pairs) == 2:
                        neighboring_two = True
        return neighboring_two

    def generate_x_stabilizers(self) -> list[dict]:
        """Subsequently builds the X stabilizers. Focus on the big `trillerpfeifen` weight-8 stabilizers."""
        x_stabilizers = []
        compatible_x_stabs = self.compatible_x_stabs()

        #check at which interface the next steane tile is placed
        adjacent_edge = self.check_interface(0)
        
        if sorted(adjacent_edge.values()) == sorted([0,4,3]):
            k = 0 #make a choice for the two possibilities 
            x_stab = {}
            x_stab.update({
                key: value
                for key, value in self.positions[0].items()
                if value in compatible_x_stabs[k]["i"]
            })
            x_stab.update({
                key: value
                for key, value in self.positions[1].items()
                if value in compatible_x_stabs[k]["i+1"]
            })
            x_stabilizers.extend((x_stab, 
                                  {key: value for key, value in self.positions[0].items() if value in {1, 2, 5, 6}},
                                  {key: value for key, value in self.positions[0].items() if value in {3, 4, 5, 6}},
                                  {key: value for key, value in self.positions[1].items() if value in {1, 2, 5, 6}},
                                  {key: value for key, value in self.positions[1].items() if value in {3, 4, 5, 6}}
                                  ))
        elif sorted(adjacent_edge.values()) == sorted([1,5,3]):
            k = 2
            x_stab = {}
            x_stab.update({
                key: value
                for key, value in self.positions[0].items()
                if value in compatible_x_stabs[k]["i"]
            })
            x_stab.update({
                key: value
                for key, value in self.positions[1].items()
                if value in compatible_x_stabs[k]["i+1"]
            })
            x_stabilizers.extend((x_stab, 
                                  {key: value for key, value in self.positions[0].items() if value in {1, 2, 5, 6}},
                                  {key: value for key, value in self.positions[0].items() if value in {0, 2, 4, 6}},
                                  {key: value for key, value in self.positions[1].items() if value in {1, 2, 5, 6}},
                                  {key: value for key, value in self.positions[1].items() if value in {3, 4, 5, 6}}
                                  ))
        else:
            msg = "Wrong edge connected between Steane patches."
            raise RuntimeError(msg)
            
        standard_steane_plaquettes = self.standard_steane()
        #remaining steane patches
        for i in range(1, len(self.positions)-1):
            current_patch = self.positions[i]
            next_patch = self.positions[i+1]
            adjacent_edge = self.check_interface(i)
            for el in compatible_x_stabs:
                #find the present stab which includes el["i"]
                stab = self.find_matching_dict(x_stabilizers, el["i"], i)
                if len(stab) == 4:
                    pos_i_new = {key: value for key, value in current_patch.items() if value in el["i"]}
                    pos_i1_new = {key: value for key, value in next_patch.items() if value in el["i+1"]}
                    neighboring_two = self.check_paired_neighbor(pos_i_new, pos_i1_new)
                    if neighboring_two:
                        x_stabilizers.remove(stab)
                        #add the weight 8 stabilizer
                        x_stab_new = {}
                        x_stab_new.update({
                            key: value
                            for key, value in current_patch.items()
                            if value in el["i"]
                        })
                        x_stab_new.update({
                            key: value
                            for key, value in next_patch.items()
                            if value in el["i+1"]
                        })
                        temp_occupied_i1 = el["i+1"] #already occupied plaquette on i+1 steane patch
                        x_stabilizers.append(x_stab_new)
                        break
            #add remaining weight 4 stabilizers on i+1
            remainder = [element for element in standard_steane_plaquettes if sorted(element) != sorted(temp_occupied_i1)]
            filtered_result = [
                {key: value for key, value in next_patch.items() if value in plaquette}
                for plaquette in remainder
            ]
            x_stabilizers.extend(filtered_result)
        
        self.x_stabilizers = x_stabilizers
        return x_stabilizers
    
    def find_matching_dict(self, x_stabilizers : list[dict], target_values: list[int], i:int) -> dict:
        """Finds teh set of stabilizers in the ith patch which have the desired target_values."""
        target_set = set(target_values)  # Convert target_values to a set for fast lookup
        temp_keys = set(self.positions[i].keys())
        x_stabilizers_temp = [
            stabilizer for stabilizer in x_stabilizers
            if not temp_keys.isdisjoint(stabilizer.keys())  # Check if there is an intersection
        ]
        candidates = []
        for stabilizer_dict in x_stabilizers_temp:
            dict_values = set(stabilizer_dict.values())  # Convert dict values to set
            # Check if all target values are present in the dict's values
            if target_set.issubset(dict_values):
                candidates.append(stabilizer_dict)
        return min(candidates, key=len)#the shortest suitable candidate

    def generate_z_stabilizers(self) -> list[dict]:
        """Builds Z stabilizers based on X stabilizers."""
        z_stabilizers = []

        #first, add all standard stabilizers
        standard_steane_plaquettes = self.standard_steane()
        for tile in self.positions:
            z_stabilizers.extend({key: value for key, value in tile.items() if value in std_stab} for std_stab in standard_steane_plaquettes)
        
        #add the weight-4, weight-2 stabilizers depending on the weight-8 X stabilizer's positions
        compatible_weight_four = self.compatible_z_stabs_weight_four()
        compatible_weight_two = self.compatible_z_stabs_weight_two()
        x_stabs_weight_eight = [stab for stab in self.x_stabilizers if len(stab) == 8]
        for i in range(len(self.positions)-1):
            current_patch = self.positions[i]
            next_patch = self.positions[i+1]
            #find weight-8 stabilizer which connects both patches
            x_stab_connect = self.find_matching_dict_z(x_stabs_weight_eight, current_patch, next_patch)
            #find the compatible weight two stab sharing qubits with the x_stab_connect
            for weight_two in compatible_weight_two:
                for key, val in current_patch.items():
                    if val == weight_two["i"]:
                        pos_current = key
                for key, val in next_patch.items():
                    if val == weight_two["i+1"]:
                        pos_next = key
                neighbors = self.neighboring_vertex(pos_current, pos_next)
                if all(value in x_stab_connect.values() for value in weight_two.values()) and neighbors:
                    #add weight_two to stabs
                    stab_temp = {}
                    for key, val in current_patch.items():
                        if val == weight_two["i"]:
                            stab_temp.update({key: val})
                    for key, val in next_patch.items():
                        if val == weight_two["i+1"]:
                            stab_temp.update({key: val})
                    z_stabilizers.append(stab_temp)
                    break
            #find the weight4 which must be disjoint to the weight 2 and pairwise neighbors on the lattice
            weight_four = self.find_disjoint_dict(current_patch, next_patch, compatible_weight_four, list(weight_two.values()))
            #add to stabs
            stab_temp = {}
            for label in weight_four["i"]:
                for key, val in current_patch.items():
                    if label == val:
                        stab_temp.update({key: val})
            for label in weight_four["i+1"]:
                for key, val in next_patch.items():
                    if label == val:
                        stab_temp.update({key: val})
            z_stabilizers.append(stab_temp)


        self.z_stabilizers = z_stabilizers
        return z_stabilizers
    
    @staticmethod
    def find_matching_dict_z(x_stabs: list[dict], current_patch: dict, next_patch: dict) -> dict:
        """Finds the x stabilizer which connects current_patch and next_patch."""
        current_keys = set(current_patch.keys())
        next_keys = set(next_patch.keys())
        
        for x_stab in x_stabs:
            x_keys = set(x_stab.keys())
            if x_keys & current_keys and x_keys & next_keys:  # Check intersection with both
                return x_stab
        return None  # Return None if no match is found
    
    def find_disjoint_dict(self, current_patch: dict, next_patch: dict, dict_list : list[dict], pair: list[int]) -> dict:
        """Finds a dictionary in which the total of integers in the values is disjoint to the given pair. AND the weight4 must bepairwise neighbors."""
        pair_set = set(pair)  # Convert the pair to a set for easy comparison
        def neighboring_pair(current_patch: dict, next_patch: dict, dictionary: dict) -> bool:
            """Checks whether at least one neighboring pair in chosen 4-weight stab."""
            for label_i in dictionary["i"]:
                for label_i1 in dictionary["i+1"]:
                    # Find positions in the dictionaries
                    positions_i = [pos for pos, label in current_patch.items() if label == label_i]
                    positions_i1 = [pos for pos, label in next_patch.items() if label == label_i1]

                    # Check if any position from i is a neighbor of a position from i+1
                    for pos_i in positions_i:
                        for pos_i1 in positions_i1:
                            if self.g.has_edge(pos_i, pos_i1):  # Check for an edge
                                return True
            return False
    
        for dictionary in dict_list:
            # Get all integers in the dictionary's values
            value_set = {val for values in dictionary.values() for val in values}
            # Check if the two sets are disjoint
            if pair_set.isdisjoint(value_set) and neighboring_pair(current_patch, next_patch, dictionary):
                return dictionary

        return None 

    def translate_checks(self) -> list[list[int]]:
        """Translates the x/z_stabilizers into check matrices."""
        #translate stabilizers in lists of global labels
        x_stabs_temp = []
        z_stabs_temp = []
        for stab in self.x_stabilizers:
            temp = [self.labels[el] for el in stab]
            x_stabs_temp.append(temp)
        for stab in self.z_stabilizers:
            temp = [self.labels[el] for el in stab]
            z_stabs_temp.append(temp)

        checks_x = []
        checks_z = []
        for stab in x_stabs_temp:
            check_temp = [0]*len(self.labels)
            for el in stab:
                check_temp[el] = 1
            checks_x.append(check_temp.copy())
        for stab in z_stabs_temp:
            check_temp = [0]*len(self.labels)
            for el in stab:
                check_temp[el] = 1
            checks_z.append(check_temp.copy())

        return checks_z, checks_x
            

    def plot_stabilizers(self, stabilizers: list[dict], size: tuple[int, int] = (7, 7)) -> None:
        """Plots the faces of the stabilizers for a given list of stabilizers (either x or z)."""
        pos = nx.get_node_attributes(self.g, "pos")
        num_faces = len(stabilizers)
        colors = cm.rainbow(np.linspace(0, 1, num_faces))  # Generate colors from the rainbow palette

        plt.figure(figsize=size)
        nx.draw(
            self.g, pos, with_labels=True, node_color="lightgray",
            edge_color="lightblue", font_size=8
        )

        for original_label, new_label in self.labels.items():
            if original_label in pos:  # Ensure the node exists in the graph
                x, y = pos[original_label]
                plt.text(
                    x, y + 0.15, str(new_label), fontsize=8,
                    color="blue", ha="center", va="center"
                )

        for face_vertices, color in zip([list(el.keys()) for el in stabilizers], colors):
            if len(face_vertices) == 2:  # Check if it's a digon
                v1, v2 = [pos[label] for label in face_vertices]
                line = Line2D([v1[0], v2[0]], [v1[1], v2[1]], color=color, lw=4)  # 'lw' is line width
                plt.gca().add_line(line)
            else:
                # Order vertices and compute the convex hull for other faces
                face_coords = convex_hull([pos[label] for label in face_vertices])
                polygon = Polygon(face_coords, closed=True, edgecolor="black", facecolor=color, alpha=0.7)
                plt.gca().add_patch(polygon)

        plt.gca().set_aspect("equal")  # Ensure the aspect ratio is equal for proper visualization
        plt.show()

def convex_hull(points: list[tuple]) -> list:
    """Find the convex hull of a set of 2D points."""
    # Sort the points by x (and by y if x's are equal)
    points = sorted(points)
    
    # Helper function: cross product of vectors OA and OB
    def cross(o: list, a: list, b: list) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build the lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build the upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove the last point of each half because it is repeated at the beginning of the other half
    return lower[:-1] + upper[:-1]

