import scipy.io
import numpy as np
import torch
class DATASET(object):
    def read_data(self):
        HC = scipy.io.loadmat('./HC.mat')['A']
        ANI = scipy.io.loadmat('./ANI.mat')['A']
        import numpy as np
        alldata = np.concatenate((HC, ANI), axis=1)
        A = np.squeeze(alldata.T)
        y1 = np.zeros(70)
        y2 = np.ones(67)
        y = np.concatenate((y1, y2), axis=0)
        series = []
        for i in range(len(A)):
            signal = A[i]
            series.append(signal)
        X = np.array(series)#(nsub,time,roi)
       # print(X.shape)
        return X, y

    def __init__(self):
        super(DATASET, self).__init__()
        X, y = self.read_data()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.n_samples = X.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.X[index], self.y[index]