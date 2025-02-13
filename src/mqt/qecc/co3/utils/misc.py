"""Misc functions for plotting and Benchmarking."""
from __future__ import annotations

import random


def generate_random_circuit(q: int, min_depth: int, tgate: bool = False, ratio: float = 0.5) -> list[tuple[int, int] | int]:
    """Random CNOT Pairs. Optional: random T gates.
    
    makes it deep enough that each qubit is used at least once
    min_depth is the minimum number of cnots
    circuit = set of terminal pairs
    the labeling does not yet follow the labels of a networkx.Graph but only range(q).

    Args:
        q (int): number of qubits of the circuit
        min_depth (int): minimal number of gates
        tgate (bool, optional): whether t gates are included or not Defaults to False.
        ratio (float, optional): ratio between t gates and cnots. 
            more t gates if smaller than 0.5. 
            note that the ratio is not deterministically fixed, only determines probabilities.
            Defaults to 0.5.

    Raises:
        ValueError: _description_

    Returns:
        list[tuple[int, int]]: _description_
    """
    if q < 2:
        msg = "q must be at least 2 to form pairs."
        raise ValueError(msg)

    pairs = [] #cnot pairs and t single qubit gate labels
    covered_elements = set()  # Keep track of elements that have appeared in a pair
    
    while len(covered_elements) < q or len(pairs) < min_depth:
        t = random.random() if tgate else 0
        if t <= ratio: 
            a, b = random.sample(range(q), 2)
            pair = (a, b)

            pairs.append(pair)

            covered_elements.update(pair)
        elif t != 1 and ratio < t <= 1:
            i = random.randrange(q)
            pairs.append(i)
            covered_elements.add(i)

    return pairs

def translate_layout_circuit(pairs: list[tuple[int, int] | int], layout: dict) -> list[tuple[tuple[int, int]] | tuple[int,int]]:
    """Translates a `pairs` circuit (with int labels) into the lattice's labels for a given layout.
    
    However, pairs does not only include tuple[int,int] but can include int as well for T gates. Then, layout will also include
    a lsit of factory positions in the key="factory_positions". but this will be ignored for this
    """
    #return [(layout[pair[0]], layout[pair[1]]) for pair in pairs]
    terminal_pairs = [(layout[pair[0]], layout[pair[1]]) if isinstance(pair, tuple) else layout[pair] for pair in pairs]
    terminal_pairs_updated = []
    for i,el in enumerate(terminal_pairs):
        if isinstance(el[0], tuple) and isinstance(el[1], tuple):
            tup1 = (int(el[0][0]), int(el[0][1]))
            tup2 = (int(el[1][0]), int(el[1][1]))
            terminal_pairs_updated.append((tup1,tup2))
        else:
            tup = (int(el[0]), int(el[1]))
            terminal_pairs_updated.append(tup)
    return terminal_pairs_updated


"""
def brute_force_qubit_assignments(data_qubit_locs: list[tuple[int, int]], num: int) -> list[dict]:
    #Bruteforces `num` qubit assignments for the given `data_qubit_locs`.
    layouts_lst = []
    counter_tot = 0
    # introduce counter and dist to get a randomized set from the permutations generator
    counter = 0
    low_lim = 1e3
    high_lim = 1e7
    dist = random.randint(low_lim, high_lim) #high threshold because permutations will give us a huge amount of cases 
    for perm in itertools.permutations(range(len(data_qubit_locs))):
        counter += 1
        if counter == dist:
            dct_temp = {}
            for i,j in zip(perm, data_qubit_locs):
                dct_temp.update({i: j})
            layouts_lst.append(dct_temp)
            counter = 0
            dist = random.randint(low_lim, high_lim)
            counter_tot += 1
        if counter_tot == num:
            break
    return layouts_lst
"""