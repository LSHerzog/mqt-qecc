"""Evaluate the routing for different Layouts."""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import mqt.qecc.co3 as co


def collect_data_space_time(instances: list[dict], hc_params: dict, reps: int, path: str) -> list[dict]:
    """Collects the data for a run which will compare space and time cost.

    Args:
        instances (list[dict]): A list of dicts which collects parameters of instances to be run.
            Each dict must contain these keys: q, t, ratio, min_depth,... also "layout_name" must be added to track the shape, since "layout_type" will be "custom".
            for generating random circuits.
        hc_params (dict): contains a value for metric, max_restarts, max_iterations, routing, optimize_factories, free_rows, parallel
        reps (int): Number of random circuits per instance (each optimized with hc and routed)
        path (str): where to store the res_lst (also intermedate save points)
        
    Returns:
        list[dict]: results dictionary for each instance (same order as instances.)
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    instances_set = {"q", "t", "min_depth", "tgate", "ratio", "custom_layout", "factory_locs", "layout_type", "layout_name"}
    for instance in instances:
        assert set(instance.keys()) == instances_set, "Wrong input for `instances`."
    hc_par_set = {"metric", "max_restarts", "max_iterations", "routing", "optimize_factories", "free_rows", "parallel", "processes"}
    assert set(hc_params.keys()) == hc_par_set, "Wrong input for `hc_params`."
    # ! layout_type for hc must be manual
    res_lst = []

    #sample circuits (should be the same for those instances for which q, min_depth, tgate, ratio) coincide
    #thus initialize with first instance's value and only change them if those values differ for the new instance
    circuits = []
    for _ in range(reps):
        circuit = co.generate_random_circuit(instances[0]["q"], instances[0]["min_depth"], instances[0]["tgate"], instances[0]["ratio"])
        circuits.append(circuit)

    for l, instance in enumerate(instances):
        logger = logging.getLogger(__name__)
        
        #check whether new values for q, min_depth, tgate, ratio. If yes sample new circuits, otherwise keep them
        if l!=0: 
            if instance["q"] == instances[l-1]["q"] and instance["min_depth"] == instances[l-1]["min_depth"] and instance["tgate"] == instances[l-1]["tgate"] and instance["ratio"] == instances[l-1]["ratio"]:
                logger.info("previous circuits kept")
            else:
                circuits = []
                for _ in range(reps):
                    circuit = co.generate_random_circuit(instance["q"], instance["min_depth"], instance["tgate"], instance["ratio"])
                    circuits.append(circuit)  
                logger.info("new circuits sampled")              


        logger.info(f"=======Instance {l}=======")
        time = []
        q = instance["q"]
        t = instance["t"]
        #min_depth = instance["min_depth"]
        #tgate = instance["tgate"]
        #ratio = instance["ratio"]
        custom_layout = instance["custom_layout"]
        factory_locs = instance["factory_locs"]   
        layout_type = instance["layout_type"]

        metric = hc_params["metric"]
        max_restarts = hc_params["max_restarts"]
        max_iterations = hc_params["max_iterations"]
        routing = hc_params["routing"]
        optimize_factories = hc_params["optimize_factories"]
        free_rows = hc_params["free_rows"]
        parallel = hc_params["parallel"]
        processes = hc_params["processes"]
        if custom_layout is not None:
            g = custom_layout[1]

        if layout_type == "custom":
            m = 5
            n = 5
            space = len(list(g.nodes()))
            #random assignments for initialization, will be overwritten by custom_layout
        else:
            raise NotImplementedError

        for circuit in circuits:
            #generate random circ
            #circuit = co.generate_random_circuit(q, min_depth, tgate, ratio)

            #do hill climbing
            hc = co.HillClimbing(
                max_restarts,
                max_iterations,
                circuit,
                layout_type,
                m,
                n,
                metric,
                factory_locs,
                len(factory_locs), # do not include different factory locations in opt
                free_rows, 
                t, 
                optimize_factories, 
                custom_layout, 
                routing
            )
            #hard coded for now
            prefix = "/mnt/c/Users/Laura/Documents/color_code_compilation/nbs-mqt-qecc/misc/"
            suffix = "test_250218"
            _, _, best_rep, score_history = hc.run(prefix, suffix, parallel, processes)

            #do the optimized routing
            input_layout = score_history[best_rep]["layout_final"]
            factory_positions = input_layout["factory_positions"]
            terminal_pairs = co.translate_layout_circuit(circuit, input_layout)
            router = co.ShortestFirstRouterTGatesDyn(m = hc.m, n = hc.n, terminal_pairs = terminal_pairs, factory_positions = factory_positions, t = t)
            if custom_layout is not None:
                router.G = g
            #update routing graph
            vdp_layers_final_dyn = router.find_total_vdp_layers_dyn()
            num_final_dyn = len(vdp_layers_final_dyn)

            #add time
            time.append(num_final_dyn)
        logger.info(f"time = {time}")
        logger.info({"space": space, "time_mean": np.mean(time), "time_std": np.std(time)})
        res_lst.append({"space": space, "time_mean": np.mean(time), "time_std": np.std(time)})
        with Path(path).open("wb") as f:
            pickle.dump(res_lst, f)

    return res_lst

    
def plot_space_time(instances: list[dict], hc_params: dict, res_lst: list[dict], path: str = "./results") -> None:
    """Plots the results from collect_data_space_time.

    Args:
        instances (list[dict]): List of instances containing various parameters.
        hc_params (dict): Hyperparameters used in the optimization.
        res_lst (list[dict]): Results containing space and time metrics.
        path (str, optional): Path to save the plot. Defaults to "./results".
    """
    assert len(instances) == len(res_lst), "instances and res_lst do not have the same length."
    colors = plt.cm.rainbow(np.linspace(0, 1, 7))

    _, ax = plt.subplots()

    # Store unique legend entries
    legend_handles = {}

    for instance, result in zip(instances, res_lst):
        space = result["space"]
        time_mean = result["time_mean"]
        time_std = result["time_std"]
        
        q = instance["q"]
        t = instance["t"]
        ratio = instance["ratio"]
        num_factories = len(instance["factory_locs"])
        layout_name = instance["layout_name"]

        # Define marker and color for each layout type
        layout_styles = {
            "hex": {"color": colors[0], "marker": "o", "label": "hex"},
            "row": {"color": colors[1], "marker": "x", "label": "row"},
            "pair": {"color": colors[2], "marker": "*", "label": "pair"},
            "sparse": {"color": colors[3], "marker": "v", "label": "sparse"},
        }

        if layout_name in layout_styles:
            style = layout_styles[layout_name]

            # Plot the point
            ax.errorbar(time_mean, space, xerr=time_std, color=style["color"], fmt=style["marker"])

            # Add label only once per layout type
            if layout_name not in legend_handles:
                legend_handles[layout_name] = Line2D(
                    [0], [0], color=style["color"], marker=style["marker"], linestyle="None", label=style["label"]
                )

        # Format the label with new lines
        text_label = f"q={q}\n t={t}\n #f={num_factories} \n ratio={ratio}"
        ax.text(time_mean, space + 0.03, text_label, fontsize=8, ha="center", va="bottom")

    # Ensure texts are not outside the plot
    y_min, y_max = ax.get_ylim() 
    ax.set_ylim(y_min, y_max + 0.03*y_max)

    # Add unique legend entries
    ax.legend(handles=list(legend_handles.values()))

    ax.set_xlabel("Time (#Layers)")
    ax.set_ylabel("Space (# logical data and ancilla qubits)")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7) 
    plt.tight_layout()

    # Create the filename based on hyperparameters
    metric = hc_params["metric"]
    max_restarts = hc_params["max_restarts"]
    max_iterations = hc_params["max_iterations"]
    file_path = Path(path) / f"space_time_metric{metric}_restarts{max_restarts}_it{max_iterations}_numinstances{len(instances)}.pdf"

    plt.savefig(file_path)
    plt.show()

