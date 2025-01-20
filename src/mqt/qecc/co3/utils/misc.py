"""Misc functions for plotting and Benchmarking."""
from __future__ import annotations

import random


def generate_random_circuit(q: int, min_depth: int) -> list[tuple[int, int]]:
    """Random CNOT Pairs.
    
    makes it deep enough that each qubit is used at least once
    min_depth is the minimum number of cnots
    circuit = set of terminal pairs
    """
    if q < 2:
        msg = "q must be at least 2 to form pairs."
        raise ValueError(msg)

    pairs = []
    covered_elements = set()  # Keep track of elements that have appeared in a pair
    
    while len(covered_elements) < q or len(pairs) < min_depth:
        a, b = random.sample(range(q), 2)
        pair = (a, b)

        pairs.append(pair)

        covered_elements.update(pair)

    return pairs

def translate_layout_circuit(pairs: list[tuple[int, int]], layout: dict) -> list[tuple[tuple[int, int]]]:
    """Translates a `pairs` circuit (with int labels) into the lattice's labels for a given layout."""
    return [(layout[pair[0]], layout[pair[1]]) for pair in pairs]

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