'''
analysis module
import convention : import maindish as md
'''
import numpy as np
import pandas as pd
from scipy.special import expit as Sigmoid
from scipy.special import softmax as Softmax

class BinaryClassification:
    
    def __init__(self, inodes, hnodes, hnumber, onodes=2, eta=0.01):
        self.inodes = inodes
        self.hnodes = hnodes
        self.hnumber = hnumber
        self.onodes = onodes
        self.eta = eta
        self.activation = lambda x : Sigmoid(x)
        self.weight = [np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))] + [np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.hnodes)) for _ in range(self.hnumber - 1)] + [np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))] 

    def select_activation(self):
        pass

    def init_weight(self):
        return [np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))] + [np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.hnodes)) for _ in range(self.hnumber - 1)] + [np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))] 

    
    def fit(self, data, target):
        '''
        data : DataFrame (preprocessed)
        target : DataFrame (applied one-hot-encoding)
        update self.weight
        return None
        '''
        data = np.array(data.values, ndmin=2, dtype=np.float64)
        target = np.array(target.values, ndmin=2, dtype=np.float64)

        for irow in range(target.shape[0]):
            idata = np.array(data[irow, :], ndmin=2).T
            itarget = np.array(target[irow, :], ndmin=2).T

            # Computing Output
            hdata = self.activation(np.dot(self.weight[0], idata))
            alldata = [idata, hdata]
            for h_num in range(1, self.hnumber):
                hdata = self.activation(np.dot(self.weight[h_num], hdata))
                alldata.append(hdata)
            output = self.activation(np.dot(self.weight[-1], hdata))
            
            # Error Backpropagation
            error = itarget - output
            herror = np.dot(self.weight[-1].T, error)
            allerror = [error, herror]
            for h_num in range(1, self.hnumber):
                herror = np.dot(self.weight[self.hnumber - h_num].T, herror)
                allerror.append(herror)
            
            # Gradient Descent
            self.weight[-1] = np.add(self.weight[-1], self.eta * np.dot((error * output * (1.0 - output)), alldata[-1].T))
            for h_num in range(1, self.hnumber+1):
                self.weight[self.hnumber - h_num] = np.add(self.weight[self.hnumber - h_num], self.eta * np.dot((allerror[h_num] * alldata[-h_num] * (1.0 - alldata[-h_num])), alldata[-h_num - 1].T))

        pass

    def predict(self, data):
        '''
        data : DataFrame (test_data)
        return DataFrame (predicted)
        '''
        data = np.array(data.values, ndmin=2, dtype=np.float64)
        predicted = pd.DataFrame(columns=['predicted_0', 'predicted_1'], index=range(data.shape[0]))

        for irow in range(data.shape[0]):
            idata = np.array(data[irow, :], ndmin=2).T

            # Computing Output
            hdata = self.activation(np.dot(self.weight[0], idata))
            for h_num in range(1, self.hnumber):
                hdata = self.activation(np.dot(self.weight[h_num], hdata))
            output = self.activation(np.dot(self.weight[-1], hdata))
            predicted.loc[irow, 'predicted_0'] = output[0,0]
            predicted.loc[irow, 'predicted_1'] = output[1,0]

        return predicted

    def accuracy_score(self, target, predicted):
        '''
        target : DataFrame
        predicted : DataFrame
        return float (accuracy)
        '''
        target = np.array(target.values, ndmin=2, dtype=np.float64)
        predicted = np.array(predicted.values, ndmin=2, dtype=np.float64)
        true = 0
        
        for irow in range(predicted.shape[0]):
            tar = np.argmax(target[irow, :])
            pred = np.argmax(predicted[irow, :])
            if tar == pred:
                true += 1
        return true / predicted.shape[0]


def main():
    '''
    # TODO
    1. HoldOut
    2. KFold
    '''
    pass

if __name__=="__main__":
    pass