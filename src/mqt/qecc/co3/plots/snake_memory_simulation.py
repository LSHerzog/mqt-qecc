from pathlib import Path

import networkx as nx
import stim
from scipy.sparse import csr_matrix
#import sinter
import numpy as np

#from stimbposd import sinter_decoders

from mqt.qecc.co3.dem_to_matrices import detector_error_model_to_check_matrices
from mqt.qecc.co3.plots.simulation_utils import get_varying_d_snakes, logicals, get_d5_different_length_sc_snakes, get_dist_three_sc_snakes
from src.mqt.qecc import CSSCode
from src.mqt.qecc.co3 import SnakeBuilderSC

if __name__ == "__main__":
    # import pymatching
    # import stim
    #
    # hx, hz, snake = get_dist_three_sc_snakes(1)
    # snake.plot_stabs(*logicals(snake, 3, hx, hz))
    # rounds = 2
    #
    # circuit = snake.snake_memory_ckt(
    #     rounds,
    #     before_round_data_depolarization=0.0,
    #     after_clifford_depolarization=0.0,
    #     before_measure_flip_probability=0.0,
    #     after_reset_flip_probability=0.0,
    # )
    # print(circuit.to_crumble_url())
    # with open('timeline-svg.svg',"w") as f:
    #     f.write(str(circuit.diagram('timeline-svg')))
    # model = circuit.detector_error_model(decompose_errors=True)
    # matching = pymatching.Matching.from_detector_error_model(model)
    # sampler = circuit.compile_detector_sampler()
    # syndrome, actual_observables = sampler.sample(shots=10000, separate_observables=True)
    # predicted_observables = matching.decode_batch(syndrome)
    # num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
    import matplotlib.pyplot as plt
    lengths = [3,5,7]
    ps = np.geomspace(0.001, 0.02, 15)
    tasks = []
    naive=False
    title = f'naive={naive}-new-varying-ds'

    for l in lengths:
        hx, hz,snake = get_varying_d_snakes(l)
        # hx,hz,snake = get_d5_different_length_sc_snakes(l)
        lx, lz = logicals(snake, l, hx, hz)
        # print("weight Z_L", len(lz))
        # print("weight X_L", len(lx))
        snake.plot_stabs(lx,lz)
        rounds = len(lx)
        print(f'rounds: {rounds} == lx weight, l is {l}')
        on = True
        for noise in ps:
            circuit = snake.snake_memory_ckt(
                rounds=rounds,
                naive=naive,
                before_round_data_depolarization=noise,
                after_clifford_depolarization=noise,
                before_measure_flip_probability=noise,
                after_reset_flip_probability=noise,
            )
            if on:
                with open(f'{title}.svg', "w") as f:
                    f.write(str(circuit.diagram('timeline-svg')))
                # print(f"smallest error stim: {len(circuit.shortest_graphlike_error())}")
                dem = detector_error_model_to_check_matrices(circuit.detector_error_model()).check_matrix
                print(f'num detectors: {circuit.num_detectors}')
                print(f'num obsbls {circuit.num_observables}')
                # plt.matshow(dem.toarray())
                # plt.show()
                # print(f"count and meas: {count_deterministic_measurements(circuit)}")

                # print(circuit.to_crumble_url())
                graph_d = len(circuit.shortest_graphlike_error())
                print(f"graph like d = {graph_d}")
                print("weight Z_L", len(lz))
                print("weight X_L", len(lx))
                print(f'check matrix shape {hz.shape}')
                on = False
                # assert(len(circuit.shortest_graphlike_error()) == len(lx)), f"{len(circuit.shortest_graphlike_error())} vs lx = {len(lx)}"
                # assert(circuit.count_determined_measurements() == circuit.num_detectors+circuit.num_observables), f"det.meas {circuit.count_determined_measurements()} vs #obsbs {circuit.num_observables} + #det {circuit.num_detectors}"
            tasks.append(
                sinter.Task(
                    circuit=circuit,
                    decoder="pymatching",
                    json_metadata={
                        "len": l,
                        "p": noise,
                        "basis": "Z",
                        "graph-d": graph_d,
                        "rounds":rounds
                    },
                )
            )
    data = sinter.collect(
        num_workers=7,
        tasks=tasks,
        max_shots=500000,
        max_errors=750,
        print_progress=True,
        save_resume_filepath=f"{title}.csv",
        decoders=['pymatching'],
    )

    # plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    sinter.plot_error_rate(
        ax=ax,
        stats=sinter.stats_from_csv_files(f"./{title}.csv"),
        x_func=lambda task: task.json_metadata["p"],
        failure_units_per_shot_func=lambda task: task.json_metadata['rounds'],
        # y_func=lambda task: task.error_rate,
        group_func=lambda task: task.json_metadata["len"],
    )
    # ps = np.geomspace(0.001, 0.01, 10)
    for k in [1]:
        ax.plot(ps, 1 - (1 - ps) ** k, label=f"Break-even k={k}", ls="--")
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"{title}")
    plt.show()
    fig.savefig(f"./{title}.png")
