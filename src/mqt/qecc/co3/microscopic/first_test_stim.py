import numpy as np
#import stimcirq

from mqt.qecc import CSSCode
from snake_builder import SnakeBuilderSTDW, SnakeBuilderSC
import networkx as nx


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

if __name__ == '__main__':

    import pymatching 
    m, n = 30,30
    G = nx.grid_2d_graph(m, n)

    # Define the position with the origin at the lower left
    pos = {(x, y): (x, y) for x, y in G.nodes()}  # Keep y as positive
    
    #d=2 #single d=3 SC patch
    #positions_rough = [
    #    [(1,0), (2,0), (3,0)],
    #    [(1,3), (2,3), (3,3)]
    #]
    #positions_smooth = [
    #    [(1,0), (1,1), (1,2), (1,3)],
    #    [(3,0), (3,1), (3,2), (3,3)]
    #]

    #d=3 #d3, 1 snake
    #positions_smooth = [
    #    [(0,1), (1,1), (2,1),(3,1)],
    #    [(5,3), (5,4), (5,5), (5,6)]
    #]
    #positions_rough = [
    #    [(0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,6), (5,6)],
    #    [(3,1), (3,2), (3,3), (4,3), (5,3)]
    #]

    d=3 #d3 and 2 snake
    positions_smooth = [
    [(1,0), (1,1), (1,2), (1,3)],
    [(3,5), (4,5), (5,5), (6,5)]
    ]
    positions_rough = [
        [(1,3), (2,3), (3,3), (3,4), (3,5)],
        [(1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (6,1), (6,2), (6,3), (6,4), (6,5)]
    ]

    #d=5
    #positions_smooth = [
    #    [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0)],
    #    [(5,13), (6,13),(7,13), (8,13), (9,13), (10,13)]
    #]
    #positions_rough = [
    #    [(0,0),(0,1), (0,2), (0,3), (0,4), (1,5), (2,6), (3,7), (4,8),(5,9),(5,10), (5,11), (5,12), (5,13)],
    #    [(5,0),(5,1), (5,2), (5,3), (5,4), (6,5), (7,6), (8,7), (9,8), (10,9), (10,10), (10,11), (10,12), (10,13)]
    #]

    #d=5 #d5, n3
    #positions_smooth = [
    #    [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0)],
    #    [(14,9), (14,10), (14,11), (14,12), (14,13), (14,14)]
    #]
    #positions_rough = [
    #    [(0,1), (0,2), (0,3), (0,4), (1,5), (2,6), (3,7), (4,8), (5,9), (6,10), (7,11), (8,12), (9,13), (10,14), (11,14), (12,14), (13,14),(14,14)],
    #    [(5,1), (5,2), (5,3), (5,4), (6,5), (7,6), (8,7), (9,8), (10,9), (11,9), (12,9), (13,9), (14,9)]
    #]

    #d=5 # d5, n4
    #positions_smooth = [
    #    [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0)],
    #    [(5,18), (6,18), (7,18), (8,18),(9,18), (10,18)]
    #]
    #positions_rough = [
    #    [(0,0),(0,1), (0,2), (0,3), (0,4), (1,5), (2,6), (3,7), (4,8),(5,9),(5,10), (5,11), (5,12), (5,13), (5,14), (5,15), (5,16), (5,17), (5,18)],
    #    [(5,0),(5,1), (5,2), (5,3), (5,4), (6,5), (7,6), (8,7), (9,8), (10,9), (10,10), (10,11), (10,12), (10,13),(10,14), (10,15), (10,16), (10,17), (10,18)]
    #]


    snake = SnakeBuilderSC(G, positions_rough, positions_smooth, d)
    _, _ = snake.create_stabs()
    print("Num Stars = ", len(snake.stars))
    print("Num Plaquettes", len(snake.plaquettes))
    print("Num Phys Qubits = ", len(snake.qubit_edges))
    print("----> Num Logical Qubits = ", len(snake.qubit_edges) - len(snake.stars) - len(snake.plaquettes))

    hx, hz, trans_dict = snake.gen_checks()
    print("Hx = ", hx.tolist())
    print("Hz = ", hz.tolist())
    # generate logical ops

    # opx, opz = logicals(snake, d, hx, hz)
    #
    # print(opz)
    # print(opx)
    # print("weight Z_L", len(opz))
    # print("weight X_L", len(opx))
    # snake.plot_stabs(opz, opx)


    ckt = snake.snake_memory_ckt(rounds=2, before_round_data_depolarization = 0.1,
                                 after_clifford_depolarization = 0.1,
                                 before_measure_flip_probability = 0.1,
                                 after_reset_flip_probability = 0.1)
    #print(ckt.to_crumble_url())
    #print(ckt.to_quirk_url())
    #cirq_circuit = stimcirq.stim_circuit_to_cirq_circuit(ckt)
    #print(cirq_circuit)
    #print(repr(ckt))
    with open("ckt2.svg", "w") as f:
        f.write(str(ckt.diagram('timeline-svg')))

    model = ckt.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    sampler = ckt.compile_detector_sampler()
    syndrome, actual_observables = sampler.sample(shots=10000, separate_observables=True)
    predicted_observables = matching.decode_batch(syndrome)
    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
    print("num errors", num_errors)