import stimcirq

from snake_builder import SnakeBuilderSTDW, SnakeBuilderSC
import networkx as nx

m=8
n=12

#generate hexagonal networkx graph
g = nx.hexagonal_lattice_graph(
    m=m, n=n, periodic=False, with_positions=True, create_using=None
)

#positions of logical qubits per triangle
positions = [
    [(1,1), (2,1), (3,1), (3,2), (2,2), (2,3), (2,4)],
    # [(4,2), (4,3), (4,4), (5,4), (5,5), (4,5), (3,5)],
    # [(3,7), (4,7), (5,7), (4,8), (5,8), (4,9), (4,10)],
    # [(6,8), (6,9), (6,10), (7,10), (7,11), (6,11), (5,11)],
    # [(7,7), (8,7), (9,7), (9,8), (8,8), (8,9), (8,10)],
    # [(10,8), (10,9), (10,10), (11,10), (11,11), (10,11),(9,11)] #!comment out adjacent lines if you want smaller snakes
]

if __name__ == '__main__':
    # d=3
    # snake = SnakeBuilderSTDW(g, positions, d)
    #
    # z_plaquettes, x_plaquettes = snake.find_stabilizers()
    #
    # size = (7,4)
    # # snake.plot_stabilizers(x_plaquettes,size)
    # # snake.plot_stabilizers(z_plaquettes,size)
    #
    # ckt = snake.snake_memory_ckt(rounds=2)
    # print(ckt.to_crumble_url())
    # print(ckt.to_quirk_url())
    # cirq_circuit = stimcirq.stim_circuit_to_cirq_circuit(ckt)
    # print(cirq_circuit)
    # print(repr(ckt))
    # with open("ckt.svg", "w") as f:
    #     f.write(str(ckt.diagram('timeline-svg')))

    # ckt2 = memory_experiment(2, CSSCode(snake.gen_check_matrix(x_plaquettes), snake.gen_check_matrix(z_plaquettes)))
    # print(ckt2.to_crumble_url())
    # print(ckt2.to_quirk_url())
    # cirq_circuit2 = stimcirq.stim_circuit_to_cirq_circuit(ckt2)
    # print(cirq_circuit2)
    # print(repr(ckt2))
    # print("other ckt")
    # print(ckt2.diagram('timeline-svg'))
    #generate check matrix
    # hz = snake.gen_check_matrix(z_plaquettes)
    # hx = snake.gen_check_matrix(x_plaquettes)
    d = 5
    # you can also switch the roles between smooth and rough boundaries
    positions_smooth = [
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        # [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6)],
        [(5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10)]
    ]
    positions_rough = [
        [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10)],
        # [(1,6), (2,7), (3,8), (4,9), (5,10)],
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        # [(1,1), (2,2), (3,3), (4,4), (5,5)]
    ]

    snake = SnakeBuilderSC(g, positions_rough, positions_smooth, d)

    ckt = snake.snake_memory_ckt(rounds=2)
    print(ckt.to_crumble_url())
    print(ckt.to_quirk_url())
    cirq_circuit = stimcirq.stim_circuit_to_cirq_circuit(ckt)
    print(cirq_circuit)
    print(repr(ckt))
    with open("ckt.svg", "w") as f:
        f.write(str(ckt.diagram('timeline-svg')))
