import innvestigate
import numpy as np
from innvestigate.analyzer import LRPZ, PatternNet, PatternAttribution, SmoothGrad, IntegratedGradients, GuidedBackprop, \
    Deconvnet

methods = {
    # "random": Random,
    # "gradient": Gradient,
    # "gradient.baseline": BaselineGradient,
    "deconvnet": Deconvnet,
    "guided_backprop": GuidedBackprop,
    "integrated_gradients": IntegratedGradients,
    "smoothgrad": SmoothGrad,

    # "lrp": LRP,
    "lrp.z": LRPZ,
    # "lrp.epsilon": LRPEpsilon,

    # "deep_taylor": DeepTaylor,
    "pattern.net": PatternNet,
    "pattern.attribution": PatternAttribution,
}
import os

os.makedirs("./results/", exist_ok=True)


def interpret(model, x_data):
    for name, method in methods.items():
        analyzer = innvestigate.create_analyzer(name, model, allow_lambda_layers=True)
        gene_analyzes = analyzer.analyze(x_data)
        gene_analyzes = np.array(gene_analyzes)
        gene_analyzes = np.reshape(gene_analyzes, newshape=(-1, x_data.shape[1]))
        print("%s has finished interpretation!" % name)
        np.savetxt(fname="./results/%s.txt" % name, X=gene_analyzes, delimiter=",")
