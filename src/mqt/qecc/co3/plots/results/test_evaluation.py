import matplotlib.pyplot as plt
import numpy as np

import mqt.qecc.co3 as co

plt.rcParams["font.family"] = "Times New Roman"

import pickle
from pathlib import Path

#from evaluation import collect_data_space_time, plot_ratio_vs_t, plot_space_time
#from layouts import gen_layout

path = "./test.pkl"

#HEX
factories_q24_hex = [(0,3), (1,8), (2,13), (7,3), (8,8), (9,13), (4,2), (5,14)]

g, data_qubit_locs, factory_ring = co.plots.gen_layout("hex", 24, factories_q24_hex)
custom_layout_q24_hex_f8 = [data_qubit_locs, g]

g, data_qubit_locs, factory_ring = co.plots.gen_layout("hex", 24, factories_q24_hex[:7])
custom_layout_q24_hex_f7 = [data_qubit_locs, g]

g, data_qubit_locs, factory_ring = co.plots.gen_layout("hex", 24, factories_q24_hex[:6])
custom_layout_q24_hex_f6 = [data_qubit_locs, g]

g, data_qubit_locs, factory_ring = co.plots.gen_layout("hex", 24, factories_q24_hex[:5])
custom_layout_q24_hex_f5 = [data_qubit_locs, g]

#ROW
factories_q24_row = [(0,3), (0,9), (0,15), (6,6), (6,12), (6,18), (5,3), (2,2)]
g, data_qubit_locs, factory_ring = co.plots.gen_layout("row", 24, factories_q24_row)
custom_layout_q24_row_f8 = [data_qubit_locs, g]

#PAIR
#factories_q24_pair = [(0,5), (0,11), (0,17), (7,3), (8,8), (8,14), (8,20), (2,2)]
#g, data_qubit_locs, factory_ring = gen_layout("pair", 24, factories_q24_pair)
#custom_layout_q24_pair = [data_qubit_locs, g]

#-----------------------------

hc_params = {
    "metric" : "crossing",
    "max_restarts": 10,
    "max_iterations": 50,
    "routing" : "dynamic",
    "optimize_factories": False,
    "free_rows" : None,
    "parallel" : True,
    "processes" : 8
}

instances = [

    #ratio 0.8
    #f=8
    {"q": 24, "t": 4, "min_depth": 24*4, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_hex_f8, "factory_locs" : factories_q24_hex, "layout_type" : "custom", "layout_name": "hex", "circuit_type": "random"},
    {"q": 24, "t": 4, "min_depth": 24*4, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_row_f8, "factory_locs" : factories_q24_row, "layout_type" : "custom", "layout_name": "row", "circuit_type": "random"},
    {"q": 24, "t": 4, "min_depth": 24*4, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_hex_f8, "factory_locs" : factories_q24_hex, "layout_type" : "custom", "layout_name": "hex", "circuit_type": "sequential"},
    {"q": 24, "t": 4, "min_depth": 24*4, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_row_f8, "factory_locs" : factories_q24_row, "layout_type" : "custom", "layout_name": "row", "circuit_type": "sequential"},
    {"q": 24, "t": 4, "min_depth": 24*4, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_hex_f8, "factory_locs" : factories_q24_hex, "layout_type" : "custom", "layout_name": "hex", "circuit_type": "parallelmax"},
    {"q": 24, "t": 4, "min_depth": 24*4, "tgate": False, "ratio": 1.0, "custom_layout": custom_layout_q24_row_f8, "factory_locs" : factories_q24_row, "layout_type" : "custom", "layout_name": "row", "circuit_type": "parallelmax"},

]

reps = 2
res_lst = co.plots.collect_data_space_time(instances, hc_params, reps, path)

with Path(path).open("rb") as f:
    res_lst = pickle.load(f)

for res in res_lst:
    print(res)

dct_mat = [] #gather for each included idx the value for t, ratio and improvement
for i, instance in enumerate(instances):
    res = res_lst[i]
    num_init_lst = res["num_init_lst"]
    num_final_lst = res["num_final_lst"]    
    improvements = []
    for ni, nf in zip(num_init_lst, num_final_lst):
        improvements.append((ni-nf)/ni)
    mean_improvement = np.mean(improvements)
    std_imrovement = np.std(improvements)
    dct_mat.append({"i": i,"mean_final_layers": np.mean(num_final_lst),"std_final_layers":np.std(num_final_lst), "mean_improvement": mean_improvement, "std_improvement": std_imrovement, "t": instance["t"], "factory_locs": instance["factory_locs"]})
for el in dct_mat:
    print(el)

#for each shape the plot
#q=24
#ratio = 0.8
#layout_name = "hex"
#min_depth = q*4
#pathp= "/home/herzog/color_code_compilation/mqt-qecc/src/mqt/qecc/co3/plots/results"
#plot_ratio_vs_t(res_lst, q, ratio, layout_name, min_depth, pathp)
