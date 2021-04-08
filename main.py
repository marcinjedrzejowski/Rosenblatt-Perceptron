import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import rosenblatt_perceptron as rp


def prepare_data(m=1000):
    X = []
    for i in range(m):
        X.append([np.random.uniform(0, 2 * np.pi), np.random.uniform(-1, 1)])
    X = np.array(X)

    Y = []
    for i, x in enumerate(X):
        if np.abs(np.sin(x[0])) > np.abs(x[1]):
            Y.append(-1)
        else:
            Y.append(1)
    return X, Y

def normalize(X):
    Xt = (X[:, 0] / np.pi) - 1
    X_norm = np.c_[Xt, X[:, 1]]
    return X_norm


if __name__ == '__main__':
    m = 1000
    np.random.seed(0)
    X, y = prepare_data(m)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10)

    plt.title("Prepared data")
    plt.xlabel("$x_{1}$")
    plt.ylabel("$x_{2}$")
    plt.show()

    X = normalize(X)

    plt.title("Normalized data without centers")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
    plt.xlabel("$x_{1}$")
    plt.ylabel("$x_{2}$")
    plt.show()

    cen = 90
    perceptron = rp.rosenblatt_perceptron(eta=1, centers=cen, sigma=0.3, k_max=5000)
    z = perceptron.gen_new_dimensions_values(X)

    centra = perceptron.get_centers_coordinates()

    plt.title("Normalized data with centers")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
    plt.scatter(centra[:, 0], centra[:, 1], c='b', s=10)
    plt.xlabel("$x_{1}$")
    plt.ylabel("$x_{2}$")
    plt.show()

    _, _ = perceptron.fit(z, y)
    print("TRAIN ACCURACY:" + str(perceptron.score(z, y)))

    z, x1, x2 = perceptron.get_dim()

    x3 = perceptron.decision_function(z).reshape(cen, cen)
    x3_2 = perceptron.class_labels_[(x3 > 0.0) * 1]


    plt.contour(x1,x2,x3_2,cmap=plt.cm.get_cmap('autumn'))
    plt.contour(x1, x2, x3_2, cmap=plt.cm.get_cmap('Dark2'))
    plt.scatter(X[:, 0],X[:, 1],s=10,c=y,cmap=plt.cm.get_cmap('Spectral'))
    plt.scatter(centra[:, 0], centra[:, 1], s=20, c='green')
    plt.xlabel("$x_{1}$")
    plt.ylabel("$x_{2}$")
    plt.title("Area graph #1")
    plt.show()

    plt.contour(x1, x2, x3, cmap=cm.seismic)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=y, cmap=plt.cm.get_cmap('Spectral'))
    plt.scatter(centra[:, 0], centra[:, 1], s=20, c='green')
    plt.xlabel("$x_{1}$")
    plt.ylabel("$x_{2}$")
    plt.title("Area graph #2")
    plt.show()

    axes = plt.figure().gca(projection='3d')
    axes.plot_surface(x1,x2,x3,cmap=cm.seismic)
    axes.set_xlabel("$x_{1}$")
    axes.set_ylabel("$x_{2}$")
    axes.set_zlabel("Weighted sum")
    plt.show()
