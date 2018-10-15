from scipy import stats
import numpy as np

# MNIST
# node_scn = np.asarray([99.22, 99.29, 99.36])
# node_scn_plus = np.asarray([99.39, 99.32, 99.24])
# edge_scn = np.asarray([])
# edge_scn_plus = np.asarray([])
# doublecapsnet = np.asarray([99.21, 99.28, 99,25])
# capsnet = np.asarray([])

mnist_data = {
    "doublecapsnet": ([99.21, 99.28, 99.25], "doublecapsnet"),
    "node_scn+": ([99.39, 99.32, 99.24], "doublecapsnet"),
    "node_scn": ([99.22, 99.29, 99.36], "doublecapsnet"),
    "egde_scn": ([99.37, 99.18, 99.26], "doublecapsnet"),
    "egde_scn+": ([99.38, 99.18, 99.26], "doublecapsnet"),
    "capsnet" :  ([99.18, 99.27, 99.09], None),
    "r-egde_scn+": ([99.33, 99.38, 99.31], "doublecapsnet"),
    "r-egde_scn+2": ([99.33, 99.38, 99.31], "node_scn+"),
}

# CIFAR 10
cifar_data = {
    "doublecapsnet": ([63.77, 64.38, 64.24], "doublecapsnet"),
    "node_scn+": ([65.98, 65.79, 66.41], "doublecapsnet"),
    "node_scn": ([65.60, 65.05, 65.17], "doublecapsnet"),
    "egde_scn": ([64.09, 64.89, 64.76], "doublecapsnet"),
    "egde_scn+": ([64.14, 63.82, 62.72], "doublecapsnet"),
    "r-egde_scn+": ([66.65, 66.51, 67.27], "doublecapsnet"),
    "r-egde_scn+2": ([66.65, 66.51, 67.27], "node_scn+"),
    "capsnet":  ([58.40, 58.97, 58.42], None),
    "r-node_scn+":  ([66.18, 64.97, 66.51], "r-egde_scn+"),
}

fashion_data = {
    "doublecapsnet": ([90.46, 90.44, 90.06], "capsnet"),
    "node_scn+": ([90.00, 90.43, 90.20], "doublecapsnet"),
    "node_scn": ([89.96, 89.89, 90.31], "doublecapsnet"),
    "egde_scn": ([90.12, 90.07, 90.26], "doublecapsnet"),
    "egde_scn+": ([90.13, 90.42, 90.18], "doublecapsnet"),
    "capsnet": ([90.02, 90.13, 90.17], None),
    "r-egde_scn+": ([90.44, 90.62, 90.40], "doublecapsnet"),
    "r-egde_scn+2": ([90.44, 90.62, 90.40], "node_scn+"),
}


def welch_dof(x, y):
    dof = (x.var() / x.size + y.var() / y.size) ** 2 / (
                (x.var() / x.size) ** 2 / (x.size - 1) + (y.var() / y.size) ** 2 / (y.size - 1))
    return int(round(dof))


def eval(data, dataset):

    print(f"###### {dataset} #####\n\n")

    for model in data.keys():

        if model == "capsnet":
            continue

        baseline_name = data[model][1]
        baseline = np.asarray(data[baseline_name][0])
        baseline_mean = baseline.mean()

        model_data = np.asarray(data[model][0])
        ttest = stats.ttest_ind(model_data, baseline, equal_var=False)
        pvalue = ttest.pvalue
        tvalue = ttest.statistic
        mean_model = model_data.mean()
        dof = welch_dof(model_data, baseline)

        print(f"\n## Scores of {model} vs {baseline_name}\n")

        print(f"Mean: {mean_model:0.2f}")
        print(f"Mean baseline: {baseline_mean:0.2f}")
        print(f"Difference: {(mean_model - baseline_mean):0.3f} +- {(model_data - baseline).std():0.3f} ")
        print(f"t value: {tvalue:0.5f}")
        print(f"p value: {pvalue:0.6f}")
        print(f"dof: {dof}")

eval(cifar_data, "CIFAR-10")
# eval(fashion_data, "Fashion-MNIST")
# eval(mnist_data, "MNIST")


