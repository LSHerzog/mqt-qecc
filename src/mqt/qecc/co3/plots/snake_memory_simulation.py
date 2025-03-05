from pathlib import Path

import networkx as nx
from scipy.sparse import csr_matrix
import sinter
import numpy as np

from stimbposd import sinter_decoders

from mqt.qecc.co3.dem_to_matrices import detector_error_model_to_check_matrices
from src.mqt.qecc import CSSCode
from src.mqt.qecc.co3 import SnakeBuilderSC


def get_dist_three_varying_snakes(length):
    if length < 1 or length >6:
        raise Exception('Length must be between 1 and 6')
    m = 10
    n = 18

    # generate hexagonal networkx graph
    g = nx.hexagonal_lattice_graph(
        m=m, n=n, periodic=False, with_positions=True, create_using=None
    )

    # positions of logical qubits per triangle
    positions = [
        # !comment out adjacent lines if you want smaller snakes
        [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (4, 2), (3, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4),
         (3, 4), (2, 4), (3, 5), (4, 5), (3, 6), (3, 7)],
        [(6, 2), (6, 3), (6, 4), (7, 4), (7, 5), (6, 5), (5, 5), (5, 6), (6, 6), (7, 6), (8, 7), (7, 7), (6, 7),
         (5, 7), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8)],
        [(4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (8, 11), (7, 11), (6, 11), (5, 11), (5, 12), (6, 12), (7, 12),
         (7, 13), (6, 13), (5, 13), (6, 14), (7, 14), (6, 15), (6, 16)],
        [(9, 11), (9, 12), (9, 13), (10, 13), (10, 14), (9, 14), (8, 14), (8, 15), (9, 15), (10, 15), (11, 16),
         (10, 16), (9, 16), (8, 16), (7, 17), (8, 17), (9, 17), (10, 17), (11, 17)],
        [(10,10), (11,10), (12,10), (13,10), (14,10), (14,11), (13,11), (12,11), (11,11), (11,12), (12,12), (13,12), (13,13), (12,13), (11,13), (12,14), (13,14), (12,15), (12,16)],
        [(15,11), (15,12), (15,13), (16,13), (16,14), (15,14), (14,14), (14,15), (15,15), (16,15), (17,16), (16,16), (15,16), (14,16), (13,17), (14,17),(15,17), (16,17), (17,17)]
    ]
    positions = positions[:length]
    d = 5
    snake = SnakeBuilderSTDW(g, positions, d)

    z_plaquettes, x_plaquettes = snake.find_stabilizers()

    size = (10, 7)
    # snake.plot_stabilizers(x_plaquettes, size)
    # snake.plot_stabilizers(z_plaquettes, size)

    # generate check matrix
    return csr_matrix(np.array(snake.gen_check_matrix(x_plaquettes))).astype(np.uint8), csr_matrix(
        np.array(snake.gen_check_matrix(z_plaquettes))).astype(np.uint8)

def get_snake_pcms_cc(d):
    if d == 3:
        m = 8
        n = 12

        # generate hexagonal networkx graph
        g = nx.hexagonal_lattice_graph(
            m=m, n=n, periodic=False, with_positions=True, create_using=None
        )

        # positions of logical qubits per triangle
        positions = [
            [(1, 1), (2, 1), (3, 1), (3, 2), (2, 2), (2, 3), (2, 4)],
            [(4, 2), (4, 3), (4, 4), (5, 4), (5, 5), (4, 5), (3, 5)],
            # [(3, 7), (4, 7), (5, 7), (4, 8), (5, 8), (4, 9), (4, 10)],
            # [(6, 8), (6, 9), (6, 10), (7, 10), (7, 11), (6, 11), (5, 11)],
            # [(7,7), (8,7), (9,7), (9,8), (8,8), (8,9), (8,10)],
            # [(10,8), (10,9), (10,10), (11,10), (11,11), (10,11),(9,11)] #!comment out adjacent lines if you want smaller snakes
        ]

        d = 3
        snake = SnakeBuilderSTDW(g, positions, d)

        z_plaquettes, x_plaquettes = snake.find_stabilizers()

        size = (7, 4)
        # snake.plot_stabilizers(x_plaquettes, size)
        # snake.plot_stabilizers(z_plaquettes, size)

        # generate check matrix
        return csr_matrix(np.array(snake.gen_check_matrix(x_plaquettes))).astype(np.uint8),csr_matrix(np.array(snake.gen_check_matrix(z_plaquettes))).astype(np.uint8),snake
    elif d == 5:
        m = 10
        n = 18

        # generate hexagonal networkx graph
        g = nx.hexagonal_lattice_graph(
            m=m, n=n, periodic=False, with_positions=True, create_using=None
        )

        # positions of logical qubits per triangle
        positions = [
            # !comment out adjacent lines if you want smaller snakes
            [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (4, 2), (3, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4),
             (3, 4), (2, 4), (3, 5), (4, 5), (3, 6), (3, 7)],
            [(6, 2), (6, 3), (6, 4), (7, 4), (7, 5), (6, 5), (5, 5), (5, 6), (6, 6), (7, 6), (8, 7), (7, 7), (6, 7),
             (5, 7), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8)],
            # [(4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (8, 11), (7, 11), (6, 11), (5, 11), (5, 12), (6, 12), (7, 12),
            #  (7, 13), (6, 13), (5, 13), (6, 14), (7, 14), (6, 15), (6, 16)],
            # [(9, 11), (9, 12), (9, 13), (10, 13), (10, 14), (9, 14), (8, 14), (8, 15), (9, 15), (10, 15), (11, 16),
            #  (10, 16), (9, 16), (8, 16), (7, 17), (8, 17), (9, 17), (10, 17), (11, 17)],
            # [(10,10), (11,10), (12,10), (13,10), (14,10), (14,11), (13,11), (12,11), (11,11), (11,12), (12,12), (13,12), (13,13), (12,13), (11,13), (12,14), (13,14), (12,15), (12,16)],
            # [(15,11), (15,12), (15,13), (16,13), (16,14), (15,14), (14,14), (14,15), (15,15), (16,15), (17,16), (16,16), (15,16), (14,16), (13,17), (14,17),(15,17), (16,17), (17,17)]
        ]

        d = 5
        snake = SnakeBuilderSTDW(g, positions, d)

        z_plaquettes, x_plaquettes = snake.find_stabilizers()

        size = (10, 7)
        # snake.plot_stabilizers(x_plaquettes, size)
        # snake.plot_stabilizers(z_plaquettes, size)

        # generate check matrix
        return csr_matrix(np.array(snake.gen_check_matrix(x_plaquettes))).astype(np.uint8),csr_matrix(np.array(snake.gen_check_matrix(z_plaquettes))).astype(np.uint8),snake
    elif d == 7:
        m = 12
        n = 16

        # generate hexagonal networkx graph
        g = nx.hexagonal_lattice_graph(
            m=m, n=n, periodic=False, with_positions=True, create_using=None
        )

        # positions of logical qubits per triangle
        positions = [
            # !comment out adjacent lines if you want smaller snakes
            [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (6, 3), (5, 3), (4, 3), (3, 3), (2, 3), (1, 3),
             (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (6, 5), (5, 5), (4, 5), (3, 5), (2, 5), (2, 6), (3, 6), (4, 6),
             (5, 6), (5, 7), (4, 7), (3, 7), (3, 8), (4, 8), (5, 8), (4, 9), (3, 9), (4, 10), (4, 11)],
            [(8, 3), (8, 4), (8, 5), (7, 5), (7, 6), (8, 6), (9, 6), (9, 7), (8, 7), (7, 7), (6, 8), (7, 8), (8, 8),
             (9, 8), (10, 9), (9, 9), (8, 9), (7, 9), (6, 9), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (10, 11),
             (9, 11), (8, 11), (7, 11), (6, 11), (5, 11), (5, 12), (6, 12), (7, 12), (8, 12), (9, 12), (10, 12),
             (11, 12)],
            # [(5, 14), (6, 14), (7, 14), (8, 14), (9, 14), (10, 14), (11, 14), (10, 15), (9, 15), (8, 15), (7, 15),
            #  (6, 15), (5, 15), (6, 16), (7, 16), (8, 16), (9, 16), (10, 16), (10, 17), (9, 17), (8, 17), (7, 17),
            #  (6, 17), (6, 18), (7, 18), (8, 18), (9, 18), (9, 19), (8, 19), (7, 19), (7, 20), (8, 20), (9, 20), (8, 21),
            #  (7, 21), (8, 22), (8, 23)],
            # [(12, 15), (12, 16), (12, 17), (11, 17), (11, 18), (12, 18), (13, 18), (13, 19), (12, 19), (11, 19),
            #  (10, 20), (11, 20), (12, 20), (13, 20), (14, 21), (13, 21), (12, 21), (11, 21), (10, 21), (10, 22),
            #  (11, 22), (12, 22), (13, 22), (14, 22), (14, 23), (13, 23), (12, 23), (11, 23), (10, 23), (9, 23), (9, 24),
            #  (10, 24), (11, 24), (12, 24), (13, 24), (14, 24), (15, 24)]
        ]

        d = 7
        snake = SnakeBuilderSTDW(g, positions, d)

        z_plaquettes, x_plaquettes = snake.find_stabilizers()

        size = (12, 9)
        # snake.plot_stabilizers(x_plaquettes, size)
        # snake.plot_stabilizers(z_plaquettes, size)

        # generate check matrix
        return csr_matrix(np.array(snake.gen_check_matrix(x_plaquettes))).astype(np.uint8),csr_matrix(np.array(snake.gen_check_matrix(z_plaquettes))).astype(np.uint8),snake
    elif d== 9:
        m = 9
        n = 18

        g = nx.hexagonal_lattice_graph(
            m=m, n=n, periodic=False, with_positions=True, create_using=None
        )

        positions = [
            [
                (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1),
                (9, 2), (8, 2), (7, 2), (6, 2), (5, 2), (4, 2), (3, 2), (2, 2),
                (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3),
                (8, 4), (7, 4), (6, 4), (5, 4), (4, 4), (3, 4), (2, 4),
                (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5),
                (7, 6), (6, 6), (5, 6), (4, 6), (3, 6),
                (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
                (7, 8), (6, 8), (5, 8), (4, 8),
                (4, 9), (5, 9), (6, 9),
                (6, 10), (5, 10), (4, 10),
                (5, 11), (6, 11),
                (5, 12),
                (5, 13)
            ],
            [
                (10, 2),
                (10, 3),
                (10, 4), (11, 4),
                (11, 5), (10, 5), (9, 5),
                (9, 6), (10, 6), (11, 6),
                (9, 7), (10, 7), (11, 7), (12, 7),
                (12, 8), (11, 8), (10, 8), (9, 8), (8, 8),
                (8, 9), (9, 9), (10, 9), (11, 9), (12, 9),
                (13, 10), (12, 10), (11, 10), (10, 10), (9, 10), (8, 10),
                (7, 11), (8, 11), (9, 11), (10, 11), (11, 11), (12, 11), (13, 11),
                (13, 12), (12, 12), (11, 12), (10, 12), (9, 12), (8, 12), (7, 12),
                (7, 13), (8, 13), (9, 13), (10, 13), (11, 13), (12, 13), (13, 13), (14, 13),
                (14, 14), (13, 14), (12, 14), (11, 14), (10, 14), (9, 14), (8, 14), (7, 14), (6, 14)
            ]
        ]
        d = 9
        snake = SnakeBuilderSTDW(g, positions, d)
        z_plaquettes, x_plaquettes = snake.find_stabilizers()
        return csr_matrix(np.array(snake.gen_check_matrix(x_plaquettes))).astype(np.uint8),csr_matrix(np.array(snake.gen_check_matrix(z_plaquettes))).astype(np.uint8),snake

    else:
        raise Exception('unsupported snake dist')

def get_dist_three_sc_snakes(nr_ancilla_patches):
    m, n = 10, 10
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

if __name__ == "__main__":
    import pymatching
    import stim

    hx, hz, snake = get_dist_three_sc_snakes(1)
    snake.plot_stabs(*logicals(snake, 3, hx, hz))
    rounds = 2

    circuit = snake.snake_memory_ckt(
        rounds,
        before_round_data_depolarization=0.0,
        after_clifford_depolarization=0.0,
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0,
    )
    print(circuit.to_crumble_url())
    with open('timeline-svg.svg',"w") as f:
        f.write(str(circuit.diagram('timeline-svg')))
    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    sampler = circuit.compile_detector_sampler()
    syndrome, actual_observables = sampler.sample(shots=10000, separate_observables=True)
    predicted_observables = matching.decode_batch(syndrome)
    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))


    # basis = "Z"
    # lengths = [1]
    # ps = np.geomspace(0.008, 0.01, 10)
    # distance = 3
    # tasks = []
    #
    # for l in lengths:
    #     hx, hz,snake = get_dist_three_sc_snakes(l)
    #     snake.plot_stabs(*logicals(snake, distance, hx, hz))
    #     rounds = distance
    #
    #     for noise in ps:
    #         circuit = snake.snake_memory_ckt(
    #             rounds,
    #             before_round_data_depolarization=noise,
    #             after_clifford_depolarization=noise,
    #             before_measure_flip_probability=noise,
    #             after_reset_flip_probability=noise,
    #         )
    #         tasks.append(
    #             sinter.Task(
    #                 circuit=circuit,
    #                 decoder="pymatching",
    #                 json_metadata={
    #                     "len": l,
    #                     "p": noise,
    #                     "basis": basis,
    #                 },
    #             )
    #         )
    # data = sinter.collect(
    #     num_workers=8,
    #     tasks=tasks,
    #     max_shots=500_000,
    #     max_errors=500,
    #     print_progress=True,
    #     save_resume_filepath=f"sc-test.csv",
    #     decoders=['pymatching'],
    # )
    #
    # # plotting
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots()
    #
    # sinter.plot_error_rate(
    #     ax=ax,
    #     stats=sinter.stats_from_csv_files("./news-test.csv"),
    #     x_func=lambda task: task.json_metadata["p"],
    #     # y_func=lambda task: task.error_rate,
    #     group_func=lambda task: task.json_metadata["len"],
    # )
    # # ps = np.geomspace(0.001, 0.01, 10)
    # for k in [1]:
    #     ax.plot(ps, 1 - (1 - ps) ** k, label=f"Break-even k={k}", ls="--")
    # # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.legend()
    # ax.set_title(f"sc-test")
    # plt.show()
