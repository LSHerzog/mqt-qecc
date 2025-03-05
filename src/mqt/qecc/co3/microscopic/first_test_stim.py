import numpy as np
import stimcirq

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
    # %%
    m, n = 10, 10
    G = nx.grid_2d_graph(m, n)

    # Define the position with the origin at the lower left
    pos = {(x, y): (x, y) for x, y in G.nodes()}  # Keep y as positive
    # %%
    d = 3
    # you can also switch the roles between smooth and rough boundaries
    # positions_smooth = [
    #    [(0,0), (0,1), (0,2), (0,3)],
    #    [(3,6), (4,6), (5,6), (6,6)]
    # ]
    # positions_rough = [
    #    [(0,3), (1,3), (2,3), (3,3), (3,4), (3,5), (3,6)],
    #    [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)]
    # ]

    # You can remove one row/col from the logical unfolded SC patches and get the same distance for X and Z,
    # this is actually cleaner
    positions_smooth = [
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(3, 5), (4, 5), (5, 5), (6, 5)]
    ]
    positions_rough = [
        [(1, 3), (2, 3), (3, 3), (3, 4), (3, 5)],
        [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5)]
    ]

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


    ckt = snake.snake_memory_ckt(rounds=2)
    print(ckt.to_crumble_url())
    print(ckt.to_quirk_url())
    cirq_circuit = stimcirq.stim_circuit_to_cirq_circuit(ckt)
    print(cirq_circuit)
    print(repr(ckt))
    with open("ckt.svg", "w") as f:
        f.write(str(ckt.diagram('timeline-svg')))
