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
        for i in range(self._num_samples):
            if self._X[i,idx]<=val:
                X_left=self._X[i]
            else:
                X_right=self._X[i]
        return X_left,X_right

    def random_sampling(self,ratio_samples):
        idx=np.random.permutation(range(self._num_samples))
        num_samples_subset=int(self._num_samples*ratio_samples)
        idx_subset=idx[:num_samples_subset]
        X_subset,y_subset=self._X[idx_subset],self._y[idx_subset]
        return Dataset(X_subset,y_subset)

    def most_frequent_label(self):
        pass