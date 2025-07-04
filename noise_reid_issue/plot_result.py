import matplotlib.pyplot as plt
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__))

range_value_snr = np.logspace(-10, 9, 30)


def plot_individual_result(result, support_size):
    fig = plt.figure()
    for i in range(3):
        plt.errorbar(
            range_value_snr,
            1 / result[:, i, 0],
            yerr=1 / result[:, i, 1],
            fmt="o",
            capsize=5,
            label=f"exp {i}",
        )
    plt.plot(range_value_snr, range_value_snr, ".", label="snr value")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(path + "/result/" + str(support_size) + "_1.png")
    plt.close(fig=fig)
    fig = plt.figure()
    for i in range(3):
        plt.errorbar(
            1 / range_value_snr,
            result[:, i, 0],
            yerr=result[:, i, 1],
            fmt="o",
            capsize=5,
            label=f"exp {i}",
        )
    plt.plot(1 / range_value_snr, 1 / range_value_snr, ".", label="snr value")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(path + "/result/" + str(support_size) + "_2.png")
    plt.close(fig=fig)


figure_global = plt.figure(figsize=(10, 6))
ax_global = plt.gca()
figure_global_divided = plt.figure(figsize=(10, 6))
ax_global_divided = plt.gca()
error_relative = []
support_size_values = np.array(range(1, 15))
for support_size in support_size_values:
    result = np.load(path + f"/result/result_{support_size}.npy").reshape(-1, 3, 2)
    plot_individual_result(result=result, support_size=support_size)
    error_relative.append([[], []])
    for i in range(1):
        ax_global.errorbar(
            1 / range_value_snr,
            result[:, i, 0],
            yerr=result[:, i, 1],
            fmt="o",
            capsize=5,
            label=f"support {support_size}",
        )
        true_value = 1 / range_value_snr
        error_relative[-1][0].append(np.abs(result[:, i, 0] - true_value) / true_value)
    for i in range(1):
        ax_global_divided.errorbar(
            1 / range_value_snr,
            result[:, i, 0] / support_size,
            yerr=result[:, i, 1] / support_size,
            fmt="o",
            capsize=5,
            label=f"support {support_size}",
        )
        true_value = support_size / range_value_snr
        error_relative[-1][1].append(np.abs(result[:, i, 0] - true_value) / true_value)
ax_global.plot(
    1 / range_value_snr, 1 / range_value_snr, "rx", label="snr value", markersize=10
)
ax_global_divided.plot(
    1 / range_value_snr, 1 / range_value_snr, "rx", label="snr value", markersize=10
)
ax_global.set_yscale("log")
ax_global.set_xscale("log")
ax_global.legend()
ax_global.set_xlim(xmax=1e12, xmin=1e-10)
ax_global.set_ylim(ymax=1e12, ymin=1e-10)
ax_global.set_title("compare to 1 / snr")
ax_global.set_xlabel("1 / snr")
ax_global.set_ylabel("estimated sigma")
figure_global.savefig(path + "/result/global.png")
ax_global_divided.set_yscale("log")
ax_global_divided.set_xscale("log")
ax_global_divided.legend()
ax_global_divided.set_xlim(xmax=1e12, xmin=1e-10)
ax_global_divided.set_ylim(ymax=1e12, ymin=1e-10)
ax_global_divided.set_title("compare to 1 / (snr * support)")
ax_global_divided.set_xlabel("1 / snr")
ax_global_divided.set_ylabel("estimated sigma / sigma")
figure_global_divided.savefig(path + "/result/global_divide.png")

error_relative = np.array(error_relative)
plt.figure()
for i, value in enumerate(range_value_snr):
    plt.plot(support_size_values, error_relative[:, 0, 0, i])
    plt.plot(support_size_values, error_relative[:, 1, 0, i])

fig = plt.figure(figsize=(10, 5))
im = plt.imshow(error_relative[:, 0, 0, :], vmax=1.0, vmin=0.01)
fig.colorbar(im)
plt.gca().set_xticks([0, 5, 10, 15, 20, 25])
plt.gca().set_xticklabels(
    [
        np.format_float_scientific(value, precision=3, exp_digits=2)
        for value in range_value_snr[[0, 5, 10, 15, 20, 25]]
    ]
)
plt.gca().set_yticks([0, 3, 5, 8, 10, 13])
plt.gca().set_yticklabels(support_size_values[[0, 3, 5, 8, 10, 13]])
plt.xlabel("signal noise ratio")
plt.ylabel("support size")
plt.title("compare to 1 / snr")
fig = plt.figure(figsize=(10, 5))
im = plt.imshow(error_relative[:, 1, 0, :], vmax=1.0, vmin=0.01)
plt.xlabel("signal noise ratio")
plt.ylabel("support size")
plt.title("compare to 1 / (snr * support)")
plt.gca().set_xticks([0, 5, 10, 15, 20, 25])
plt.gca().set_xticklabels(
    [
        np.format_float_scientific(value, precision=3, exp_digits=2)
        for value in range_value_snr[[0, 5, 10, 15, 20, 25]]
    ]
)
plt.gca().set_yticks([0, 3, 5, 8, 10, 13])
plt.gca().set_yticklabels(support_size_values[[0, 3, 5, 8, 10, 13]])
fig.colorbar(im)
plt.show()
