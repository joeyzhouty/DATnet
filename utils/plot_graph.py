import numpy as np
import matplotlib.pyplot as plt

# twitter ratio 16, 54, 2
# spanish ratio 68, 89, 2
# spanish ext low 15, 81, 5
# twitter ext low 0, 46, 5
# dutch ext low 5, 75, 5


def plot_part_graph(features, title="", start=16, end=51, step=2, xlabel="Target Dataset Ratio", ylabel="F1 Score (%)",
                    format="pdf"):
    font = {'fontname': "Times New Roman"}
    base, adv, trans_p, trans_f, adv_trans_p, adv_trans_f = features[1, 0:], features[2, 0:], features[3, 0:], \
                                                            features[4, 0:], features[5, 0:], features[6, 0:]
    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(left=0.11, right=0.98, bottom=0.1, top=0.98)
    plt.style.use('classic')
    plt.xlim(0.8, features.shape[1] + 0.2)
    plt.ylim(start, end)
    plt.xlabel(xlabel, fontsize=40, **font)
    plt.ylabel(ylabel, fontsize=40, **font)
    group_labels = [str(x) for x in features[0, 0:]]
    # group_labels = [str(int(x)) for x in features[0, :]]
    x = np.linspace(1, features.shape[1], features.shape[1])
    plt.xticks(x, group_labels, rotation=0, fontsize=34, **font)
    plt.yticks(np.arange(start, end, step=step), fontsize=34, **font)
    plt.grid(which="both", axis="both")
    # marker style -- https://matplotlib.org/api/markers_api.html
    plt.plot(x, base, ':c', marker='o', label='Base', alpha=0.8, linewidth=5, markersize=16)
    plt.plot(x, adv, ":g", marker="*", label="Base + AT", alpha=0.8, linewidth=5, markersize=18)
    plt.plot(x, trans_f, "--y", marker="v", label="F-Transfer (GRAD)", alpha=0.8, linewidth=5, markersize=16)
    plt.plot(x, trans_p, "-m", marker="D", label="P-Transfer (GRAD)", alpha=0.8, linewidth=5, markersize=16)
    plt.plot(x, adv_trans_f, "--b", marker="^", label="DATNet-F", alpha=0.8, linewidth=5, markersize=16)
    plt.plot(x, adv_trans_p, '-r', marker='s', label='DATNet-P', alpha=0.8, linewidth=5, markersize=16)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.legend(loc=4, fontsize=36)
    plt.savefig('{}.{}'.format(title, format), format=format)
    plt.show()


data = np.genfromtxt("data/twitter_ratio.csv", delimiter=",")
plot_part_graph(data, title="Twitter_Ratio", start=16, end=54, step=2, format="pdf")
