import numpy as np

class Dataset:
    def __init__(self,X:np.array,y:np.array):
        self._X=X
        self._y=y
        self._num_samples, self._num_features=X.shape

    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_samples(self):
        return self._num_samples

    @property
    def num_features(self):
        return self._num_features

    def split(self,idx,val):
        X_left = []
        y_left = []
        X_right = []
        y_right = []
        for i in range(self._num_samples):
            if self._X[i,idx]<=val:
                X_left.append(self._X[i])
                y_left.append(self._y[i])
            else:
                X_right.append(self._X[i])
                y_right.append(self._y[i])

        X_left = np.vstack(X_left) if X_left else np.empty((0, self._X.shape[1]))
        X_right = np.vstack(X_right) if X_right else np.empty((0, self._X.shape[1]))
        y_left = np.array(y_left)
        y_right = np.array(y_right)
        return Dataset(X_left,y_left), Dataset(X_right,y_right)

    def random_sampling(self,ratio_samples):
        idx=np.random.permutation(range(self._num_samples))
        num_samples_subset=int(self._num_samples*ratio_samples)
        idx_subset=idx[:num_samples_subset]
        X_subset,y_subset=self._X[idx_subset],self._y[idx_subset]
        return Dataset(X_subset,y_subset)

    def most_frequent_label(self):
        pass