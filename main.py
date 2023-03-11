import numpy as np
import matplotlib.pyplot as plt

def projections_onto_orientation(pcm, ncm, X):

    projX = []
    # calculate orientation line
    orientation = pcm - ncm
    orientation_normal = orientation / np.linalg.norm(orientation)
    #print(f'n: {orientation_normal}')
    for i in range(len(X)):
        projX.append(np.dot(X[i], orientation_normal))
    return projX

class LinearClassifier:

    def __init___(self):
        self.best_sep = np.array([])
        self.pcm = 0  # positive centre mean
        self.ncm = 0  # negative centre mean
        self.polarity = 1
        self.alpha = 0

    def find_sep_line(self, X, y, w):

        pc = X[1::2]  # positives
        nc = X[::2]  # negatives
        # Calculate mean of the classes
        mean_w = np.reshape(w, [-1, 1])  # reshape weights in order to perform the calculations
        pc = sum(pc*mean_w[1::2])
        self.pcm = pc / sum(mean_w[1::2])
        nc = sum(nc*mean_w[::2])
        self.ncm = nc / sum(mean_w[::2])

        print(f'Weighted +1 class mean: {self.pcm}')
        print(f'Weighted -1 class mean: {self.ncm}')

        projX = []
        projX = projections_onto_orientation(self.pcm, self.ncm, X_train)
        #print(f'projections: {projX}')

        #plt.axline(self.pcm, self.ncm, color='gray', linestyle='dashed')
        #plt.show()

        # sort projX but also w and y
        projX, y, w = zip(*sorted(zip(projX, y, w)))
        canditate_points = np.concatenate((np.expand_dims(projX, axis=1), np.expand_dims(w, axis=1), np.expand_dims(y, axis=1)), axis=1)

        sep_arr = np.array([])
        error_arr = np.array([])
        pol = 1

        # find all separation lines and their error
        for i in range(len(canditate_points) - 1):

            prediction1 = np.ones(len(y)) * (-1)
            prediction2 = np.ones(len(y))
            # calculate separation point
            sep_line = (canditate_points[i + 1][0] + canditate_points[i][0]) / 2
            sep_arr = np.append(sep_arr, sep_line)

            # classify points
            for j in range(len(canditate_points)):
                if canditate_points[j][0] < sep_line:
                    prediction1[j] = 1
                    prediction2[j] = -1

            error = np.array([sum(canditate_points[:, 1] * (np.not_equal(canditate_points[:, 2], prediction1))),
                              sum(canditate_points[:, 1] * (np.not_equal(canditate_points[:, 2], prediction2)))])
            error_arr = np.append(error_arr, error)

        error_arr = np.reshape(error_arr, (-1, 2))
        min_error = float('inf')
        line_i = 0
        # get minimum error and polarity
        for err1, err2 in error_arr:
            if err1 < min_error:
                min_error = err1
                self.polarity = 1
                self.best_sep = sep_arr[line_i]
            if err2 < min_error:
                min_error = err2
                self.polarity = -1
                self.best_sep = sep_arr[line_i]

            line_i +=1
        return min_error

    def predict(self, X):

        y_pred = np.zeros(len(X))

        projections = projections_onto_orientation(self.pcm, self.ncm, X)
        #print(f'len projections: {len(projections)}')
        #print(f'len X: {len(X)}')
        # print(f'projections: {projections}')
        # print(f'in predict polarity: {self.polarity}')
        for i in range(len(X)):
            if self.polarity == 1:
                if projections[i] < self.best_sep:
                    y_pred[i] = 1
                else:
                    y_pred[i] = -1
            else:
                if projections[i] < self.best_sep:
                    y_pred[i] = -1
                else:
                    y_pred[i] = 1
            """if projections[i] < best_separation_line:
                y_pred[i] = 1
            else:
                y_pred[i] = -1"""
        return y_pred


class Adaboost:

    def __init__(self, n_clsfs):
        self.clsfs = []
        self.n_clsfs = n_clsfs
        self.alphas = []

    def fit(self, X, y):


        # init weights
        w = np.ones(len(y)) * 1 / len(y)

        for i in range(0, self.n_clsfs):

            linear = LinearClassifier()
            min_error = linear.find_sep_line(X, y, w)
            print(f'min_error: {min_error}')

            # calculate predictions
            predictions = linear.predict(X)


            # calculate alpha
            EPS = 1e-10  # a small non-zero value
            alpha = 0.5 * np.log((1 - min_error) / min_error)
            print(f'alpha: {alpha}')

            # update weights
            w = w * np.exp(-alpha * y * predictions)
            w = w / sum(w)
            #print(f'w: {w}')

            # save classifier
            self.alphas.append(alpha)
            self.clsfs.append(linear)

    def predict(self, X):
        clf_preds = []
        for i in range(0, self.n_clsfs):
            clf_preds.append(self.alphas[i] * self.clsfs[i].predict(X))

        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred).astype(int)

        return y_pred


def load_data():
    # load data
    data = np.loadtxt("adaboost-train-22.txt")
    X_train = data[:, 0:2]
    y_train = data[:, 2]
    data2 = np.loadtxt("adaboost-test-22.txt")
    X_test = data2[:, 0:2]
    y_test = data2[:, 2]

    return X_train, y_train, X_test, y_test

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# main method
if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_data()

    # train set
    acc_list = []
    for i in range(1, 40):  # iterate through diffrent numbers of weak classifiers

        # initialize adaboost object with i weak classifiers
        adaboost = Adaboost(i)
        # train
        adaboost.fit(X_train, y_train)
        # predict using strong classifier
        predictions = adaboost.predict(X_train)
        # save accuracies so we can plot them after
        acc = accuracy(y_train, predictions)
        acc_list.append(acc)


    acc_list2 = []

    # test set
    for i in range(1, 100):
        adaboost = Adaboost(i)
        adaboost.fit(X_train, y_train)
        predictions = adaboost.predict(X_test)
        acc = accuracy(y_test, predictions)
        acc_list2.append(acc)

    # plotting results
    print(f"Accuracy list: {acc_list2}")
    plt.plot(acc_list)
    plt.xlabel('weak classifiers')
    plt.ylabel('accuracy')
    plt.show()

    plt.plot(acc_list2)
    plt.xlabel('weak classifiers')
    plt.ylabel('accuracy')
    plt.show()

    print(f"Training accuracy list: {acc_list}")
    print(f"Achieved 100% accuracy on training set in {acc_list.index(1.0)} steps")

    print(f"Testing accuracy list: {acc_list2}")
    print(f"Achieved 100% accuracy on training set in {acc_list2.index(1.0)} steps")