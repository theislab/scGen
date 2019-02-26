import innvestigate
from innvestigate.analyzer import LRPZ, LRPEpsilon, IntegratedGradients

from scgen.interpret.IntegratedGradients import *

methods = {
    # "lrp.z": LRPZ,
    # "lrp.epsilon": LRPEpsilon,
    "integrated_gradients": IntegratedGradients,
}
import os
os.makedirs("./results/", exist_ok=True)


def interpret(model, x_data, neuron_idx=None, save=True):
    for name, method in methods.items():
        analyzer = innvestigate.create_analyzer(name, model, allow_lambda_layers=True)
        gene_analyzes = analyzer.analyze(x_data)
        gene_analyzes = np.array(gene_analyzes)
        gene_analyzes = np.reshape(gene_analyzes, newshape=(-1, x_data.shape[1]))
        print(f"{name} for neuron-{neuron_idx} has finished interpretation!")
        if save:
            if neuron_idx is not None:
                np.savetxt(fname=f"./results/{name}_train_{neuron_idx}.txt", X=gene_analyzes, delimiter=",")
            else:
                np.savetxt(fname=f"./results/{name}_train_.txt", X=gene_analyzes, delimiter=",")
        return gene_analyzes
