import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def img2d(data, xlabel="lons", ylabel="lats", title="", vmin=None, vmax=None):
    if title != "":
        plt.title(title)
    plt.imshow(data, cmap="rainbow", vmin=vmin, vmax=vmax)
    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.set_ylabel(ylabel)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.colorbar()


def scatter2d(data, xidx=0, yidx=1, zidx=-1, axes=None, xlabel="lons", ylabel="lats", title=""):
    if title != "":
        plt.title(title)
    if axes is None:
        if zidx >= 0:
            scatter = plt.scatter(data[:, xidx], data[:, yidx], s=data[:, zidx], alpha=0.5)
        else:
            scatter = plt.scatter(data[:, xidx], data[:, yidx], alpha=0.5)
    else:
        if zidx >= 0:
            scatter = axes.scatter(data[:, xidx], data[:, yidx], s=data[:, zidx], alpha=0.5)
        else:
            scatter = axes.scatter(data[:, xidx], data[:, yidx], alpha=0.5)
    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.set_ylabel(ylabel)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    return scatter


def scatter3d(data, axes=None, xlabel='h', ylabel='angle', zlabel='var', alpha=0.5, title=""):
    if title != "":
        plt.title(title)
    if axes is None:
        axes = plt.axes(projection='3d')
    X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
    scatter = axes.scatter(X, Y, Z, alpha=alpha)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_zlabel(zlabel)
    return scatter
