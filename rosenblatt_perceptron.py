from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import math as mt

class rosenblatt_perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=1.0, centers=10, sigma=0.1, k_max=1000):
        self.class_labels_ = None
        self.eta_ = eta
        self.centers_ = centers
        self.w_ = None
        self.k_ = None
        self.sigma_ = sigma
        self.centers_cord_ = None
        self.k_max_ = k_max

    def gen_feats_matrix(self, new_matrix, X, Y):
        for i in range(len(new_matrix)):
            for j in range(len(new_matrix[i])):
                new_matrix[i, j] = self.centres_distance(X[i], Y[j])
        return new_matrix

    def get_centers_coordinates(self):
        return self.centers_cord_

    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        y_normalized = np.ones(m)
        y_normalized[y == self.class_labels_[0]] = -1
        X = np.c_[np.ones(m), X]
        self.k_=0
        self.w_ = np.zeros(n + 1)
        while True:
            E = []  # przykłady błędnie sklasyfikowane
            E_y = []
            for i in range(m):
                x = X[i]
                s = self.w_.dot(x)
                f = -1 if s <= 0 else 1
                if f != y_normalized[i]:
                    E.append(x)
                    E_y.append(y_normalized[i])
            if len(E) == 0:
                break
            i = int(np.random.rand() * len(E))
            x = E[i]
            self.w_ = self.w_ + self.eta_ * E_y[i] * x
            self.k_ += 1
            if self.k_ >= self.k_max_:
                break
        return self.w_, self.k_


    def predict(self, X):
        return self.class_labels_[(self.decision_function(X) > 0.0) * 1]

    def decision_function(self, X):
        m = X.shape[0]
        return self.w_.dot(np.c_[np.ones(m), X].T)

    def centres_distance(self, x, c):
        return mt.exp(- ((((x[0] - c[0]) ** 2) + ((x[1] - c[1]) ** 2)) / (2 * (self.sigma_ ** 2))))

    def gen_new_dimensions_values(self, X):
        A = np.zeros((X.shape[0], self.centers_))

        random_centres = []
        for i in range(self.centers_):
            random_centres.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
        random_centres = np.array(random_centres)
        self.centers_cord_ = random_centres

        A = self.gen_feats_matrix(A, X, random_centres)

        return A

    def get_dim(self):
        centers = self.centers_
        x = np.linspace(-1, 1, centers)
        x1, x2 = np.meshgrid(x, x)
        new_feats = np.stack((x1, x2), axis=2)
        new_feats = new_feats.reshape(new_feats.shape[0]**2, 2)
        z = np.zeros((new_feats.shape[0], centers))
        z = self.gen_feats_matrix(z, new_feats, self.centers_cord_)
        return z, x1, x2
