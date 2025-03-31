from pathlib import Path

import networkx as nx
import sinter
import stim
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from sinter._plotting import plot_custom

from src.mqt.qecc import CSSCode
from src.mqt.qecc.co3 import SnakeBuilderSC


def get_dist_three_sc_snakes(nr_ancilla_patches):
    m, n = 15, 15
    G = nx.grid_2d_graph(m, n)

    # Define the position with the origin at the lower left
    pos = {(x, y): (x, y) for x, y in G.nodes()}  # Keep y as positive
    d = 3
    if nr_ancilla_patches == 1:
        positions_smooth = [
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(5, 3), (5, 4), (5, 5), (5, 6)]
        ]
        positions_rough = [
            [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 6), (5, 6)],
            [(3, 1), (3, 2), (3, 3), (4, 3), (5, 3)]
        ]

    elif nr_ancilla_patches == 2:
        positions_smooth = [
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(3, 8), (4, 8), (5, 8), (6, 8)]
        ]
        positions_rough = [
            [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (3, 7), (3, 8)],
            [(3, 1), (3, 2), (3, 3), (4, 4), (5, 5), (6, 6), (6, 7), (6, 8)]
        ]
    elif nr_ancilla_patches == 3:
        positions_smooth = [
            [(1, 1), (2, 1), (3, 1), (4, 1)],
            [(9, 6), (9, 7), (9, 8), (9, 9)]
        ]
        positions_rough = [
            [(1, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 9), (9, 9)],
            [(4, 1), (4, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 6), (9, 6)]
        ]
    elif nr_ancilla_patches == 4:
        positions_smooth = [
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(3, 11), (4, 11), (5, 11), (6, 11)]
        ]
        positions_rough = [
            [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11)],
            [(3, 1), (3, 2), (3, 3), (4, 4), (5, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11)]
        ]

    else:
        raise ValueError(f'unsupported #ancilla patches {nr_ancilla_patches}')
    snake = SnakeBuilderSC(G, positions_rough, positions_smooth, d)
    _, _ = snake.create_stabs()
    hx, hz, trans_dict = snake.gen_checks()
    return hx, hz, snake

def logicals(snake, d, hx, hz):
    code = CSSCode(distance=d, Hx=hx, Hz=hz)
    print(code.Lx, np.sum(code.Lx))
    print(code.Lz, np.sum(code.Lz))

    assert len(code.Lx) == 1 and len(code.Lz) ==1, "More than one qubit encoded!"

    #translate Lz into list of edges on the graph
    trans_dict_rev = {value: key for key, value in snake.trans_dict.items()}
    opz = []
    for i, el in enumerate(code.Lz[0]):
        if el == 1:
            opz.append(trans_dict_rev[i])
    opx = []
    for i, el in enumerate(code.Lx[0]):
        if el == 1:
            opx.append(trans_dict_rev[i])
    return opx, opz

def check_matchable(h):
    """checks whether max 2 nonzero entries per col"""
    num_rows,  num_cols = np.shape(h)
    for i in range(num_cols):
        col = h[:,i]
        num_nonzero = np.sum(col)
        #print(num_nonzero)
        if num_nonzero > 2:
            print("Not matchable!")
            return
    print("Matchable.")

def get_varying_distance_len_3_sc_snakes(d):
    if d == 3:
        m, n = 15, 15
        G = nx.grid_2d_graph(m, n)
        positions_smooth = [
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(3, 11), (4, 11), (5, 11), (6, 11)]
        ]
        positions_rough = [
            [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11)],
            [(3, 1), (3, 2), (3, 3), (4, 4), (5, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11)]
        ]
    elif d ==5:
        m, n = 20, 20
        G = nx.grid_2d_graph(m, n)

        # Define the position with the origin at the lower left
        pos = {(x, y): (x, y) for x, y in G.nodes()}  # Keep y as positive

        # plt.figure(figsize=(12,12))
        # nx.draw(G, pos, with_labels=True)
        # %%
        d = 5

        positions_smooth = [
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
            [(5, 18), (6, 18), (7, 18), (8, 18), (9, 18), (10, 18)]
        ]
        positions_rough = [
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (5, 10), (5, 11), (5, 12),
             (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18)],
            [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9), (10, 10), (10, 11),
             (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18)]
        ]
    else:
        raise ValueError(f'unsupported distance {d}')
    snake = SnakeBuilderSC(G, positions_rough, positions_smooth, d)
    _, _ = snake.create_stabs()
    hx, hz, trans_dict = snake.gen_checks()
    return hx, hz, snake

def count_deterministic_measurements(measurement_loop: stim.Circuit):
    result = 0
    measurements = []
    simulator = stim.TableauSimulator()

    # Do a few iterations to get into the steady state.
    simulator.do(measurement_loop * 10)

    # Do an iteration counting the number of determined measurements.
    for instruction in measurement_loop:
        # TODO: recurse into sub-loops if needed
        assert isinstance(instruction, stim.CircuitInstruction)
        # TODO: generalize this to work for all measurement operations.
        assert instruction.name not in ["MX", "MY", "MRX", "MRY", "MPP"]

        if instruction.name in ["M", "MR"]:
            for gate_target in instruction.targets_copy():
                assert gate_target.is_qubit_target
                is_random = simulator.peek_z(gate_target.value) == 0
                if not is_random:
                    result += 1
                else:
                    measurements.append(gate_target)
        simulator.do(instruction)

    return result, measurements

def get_d5_different_length_sc_snakes(nr_anc_patches):
    # necessary lattice for all instances
    m, n = 23, 23
    G = nx.grid_2d_graph(m, n)
    pos = {(x, y): (x, y) for x, y in G.nodes()}  # Keep y as positive
    if nr_anc_patches == 1:
        # ------------------n=1 + 2 logical------------------
        positions_smooth = [
            [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
            [(9, 10), (9, 9), (9, 8), (9, 7), (9, 6), (9, 5)]
        ]
        positions_rough = [
            [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5)],
            [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 10), (7, 10), (8, 10),
             (9, 10)]
        ]
    elif nr_anc_patches == 2:
        # ------------------n=2 + 2 logical------------------
        positions_smooth = [
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
            [(5, 13), (6, 13), (7, 13), (8, 13), (9, 13), (10, 13)]
        ]
        positions_rough = [
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (5, 10), (5, 11), (5, 12),
             (5, 13)],
            [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9), (10, 10), (10, 11), (10, 12),
             (10, 13)]
        ]
    elif nr_anc_patches == 3:
        # ------------------n=3 + 2 logical------------------
        positions_smooth = [
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
            [(14, 9), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14)]
        ]
        positions_rough = [
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11), (8, 12), (9, 13),
             (10, 14), (11, 14), (12, 14), (13, 14), (14, 14)],
            [(5, 1), (5, 2), (5, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9), (11, 9), (12, 9), (13, 9), (14, 9)]
        ]
    elif nr_anc_patches == 4:
        # ------------------n=4 + 2 logical------------------
        positions_smooth = [
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
            [(5, 18), (6, 18), (7, 18), (8, 18), (9, 18), (10, 18)]
        ]
        positions_rough = [
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (5, 10), (5, 11), (5, 12),
             (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18)],
            [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9), (10, 10), (10, 11), (10, 12),
             (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18)]
        ]
    elif nr_anc_patches == 5:
        # ------------------n=5 + 2 logical------------------
        positions_smooth = [
            [(x, 0) for x in range(6)],
            [(19, x) for x in range(14, 20)]
        ]
        positions_rough = [
            [(0, x) for x in range(5)] + [(1 + x, 5 + x) for x in range(15)] + [(x, 19) for x in range(15, 20)],
            [(5, x) for x in range(5)] + [(6 + x, 5 + x) for x in range(10)] + [(x, 14) for x in range(15, 20)]
        ]
    else:
        raise ValueError(f'unsupported #ancilla patches {nr_anc_patches}')
    snake = SnakeBuilderSC(g=G,
                           positions_rough=positions_rough,
                           positions_smooth=positions_smooth,
                           d=5)
    _, _ = snake.create_stabs()
    hx, hz, trans_dict = snake.gen_checks()
    return hx, hz, snake

def get_varying_d_snakes(distance):
    # inputs for fixed shape and different d of the sep.patches
    if distance == 3:
        # -------------------d=3-------------------------
        m, n = 10, 10
        G = nx.grid_2d_graph(m, n)
        pos = {(x, y): (x, y) for x, y in G.nodes()}

        d = 3
        positions_smooth = [
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(3, 8), (4, 8), (5, 8), (6, 8)]
        ]
        positions_rough = [
            [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (3, 7), (3, 8)],
            [(3, 1), (3, 2), (3, 3), (4, 4), (5, 5), (6, 6), (6, 7), (6, 8)]
        ]
    elif distance == 5:
        # -------------------d=5-------------------------
        m, n = 20, 20
        G = nx.grid_2d_graph(m, n)
        pos = {(x, y): (x, y) for x, y in G.nodes()}

        d = 5
        positions_smooth = [
            [(5, 14), (6, 14), (7, 14), (8, 14), (9, 14), (10, 14)],
            [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
        ]
        positions_rough = [
            [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (5, 11), (5, 12), (5, 13),
             (5, 14)],
            [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (10, 11), (10, 12), (10, 13),
             (10, 14)]
        ]
    elif distance == 7:
        # -------------------d=7-------------------------
        m, n = 30, 30
        G = nx.grid_2d_graph(m, n)
        pos = {(x, y): (x, y) for x, y in G.nodes()}

        d = 7
        positions_smooth = [
            [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
            [(7, 20), (8, 20), (9, 20), (10, 20), (11, 20), (12, 20), (13, 20), (14, 20)]
        ]
        positions_rough = [
            [(7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13),
             (14, 14), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20)],
            [(7, 20), (7, 19), (7, 18), (7, 17), (7, 16), (7, 15), (7, 14), (6, 13), (5, 12), (4, 11), (3, 10), (2, 9),
             (1, 8), (0, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1)]
        ]
    elif distance == 9:
        # -------------------d=9-------------------------
        m, n = 40, 40
        G = nx.grid_2d_graph(m, n)
        pos = {(x, y): (x, y) for x, y in G.nodes()}

        d = 9
        positions_smooth = [
            [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1)],
            [(9, 26), (10, 26), (11, 26), (12, 26), (13, 26), (14, 26), (15, 26), (16, 26), (17, 26), (18, 26)]
        ]
        positions_rough = [
            [(9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13),
             (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (18, 24),
             (18, 25), (18, 26)],
            [(9, 26), (9, 25), (9, 24), (9, 23), (9, 22), (9, 21), (9, 20), (9, 19), (9, 18), (8, 17), (7, 16), (6, 15),
             (5, 14), (4, 13), (3, 12), (2, 11), (1, 10), (0, 9), (0, 8), (0, 7), (0, 6), (0, 5), (0, 4), (0, 3), (0, 2),
             (0, 1)]
        ]
    elif distance == 11:
        # -------------------d=11-------------------------

        m, n = 40, 40
        G = nx.grid_2d_graph(m, n)
        pos = {(x, y): (x, y) for x, y in G.nodes()}

        d = 11
        positions_smooth = [
            [(x, 1) for x in range(12)],
            [(x, 32) for x in range(11, 23)]
        ]
        positions_rough = [
            [(11, x) for x in range(1, 12)] + [(x, x) for x in range(11, 23)] + [(22, x) for x in range(22, 33)],
            [(0, x) for x in range(1, 12)] + [(x, x + 11) for x in range(12)] + [(11, x) for x in range(22, 33)]
        ]
    else:
        raise ValueError(f'unsupported distance {distance}')
    snake = SnakeBuilderSC(G, positions_rough, positions_smooth, distance)
    _, _ = snake.create_stabs()
    hx, hz, trans_dict = snake.gen_checks()
    return hx, hz, snake


def plot_file(filename):

    fig, ax = plt.subplots()

    sinter.plot_error_rate(
        ax=ax,
        stats=sinter.stats_from_csv_files(f"./{filename}.csv"),
        x_func=lambda task: task.json_metadata["p"],
        failure_units_per_shot_func=lambda stats: stats.json_metadata['rounds'],
        group_func=lambda task: task.json_metadata["len"],
    )
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"{filename}")
    plt.show()

def plot_projected_d(filename):

    fig, ax = plt.subplots()

    plot_custom(
        ax=ax,
        stats=sinter.stats_from_csv_files(f"./{filename}.csv"),
        x_func=lambda task: task.json_metadata["len"],
        y_func=lambda task: task.errors/task.shots,
        filter_func=lambda task: task.json_metadata["p"] < 0.009,
        group_func=lambda task: task.json_metadata["p"],
    )
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"X logical error rate per shot vs distance")
    plt.show()

if __name__ == '__main__':
    plot_projected_d('naive=False-new-varying-lens')
