from dataset import *

float_formatter = "{:.2E}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
from datetime import datetime

def normalize(a):
    col_sums = a.sum(axis=0)
    new_matrix = a / col_sums[np.newaxis, :]
    return new_matrix

class Shaman():
    def __init__(self, dataset, standardize = True):
        self.p = dataset.p
        self.dataset = dataset
        self.thetas = np.zeros([self.p + 1], dtype = float)
        self.old_thetas = self.thetas
        self.lr = 1.0
        self.lr_decay = 1.0 / 2
        self.oldcost = 0.0
        self.c = 1.0 / 2
        self.lr_increase = 1.5
        self.start_time = None
        self.time_limit = 2
        self.standardize = standardize

    def time_stop(self):
        return ((datetime.now() - self.start_time).seconds >= 2)

    def predict(self, data, thetas = None):
        if thetas is None:
            thetas = self.thetas
        return (np.dot(data, thetas))


    def error(self, thetas = None):
        if thetas is None:
            thetas = self.thetas
        predictions = self.predict(self.dataset.x, thetas)
        error = predictions - self.dataset.y
        return (error)


    def mean_squared_error(self, thetas = None):
        if thetas is None:
            thetas = self.thetas   
        squared_error = np.square(self.error(thetas))
        return (np.mean(squared_error) / 2)


    def compute_gradients(self):
        error = self.error()
        gradients = np.dot(error, self.dataset.x)
        gradients = gradients / len(self.dataset.y)
        # gradients = gradients        
        return (gradients)


    def ajimo_goldstein_condition(self, l2_grad_squared, gradients, lr):
        thetas = self.thetas - lr * gradients
        cost = self.mean_squared_error(thetas)
        objective = self.newcost - (self.c * lr * l2_grad_squared)
        return (cost <= objective)


    def ajimo(self, gradients):
        l2_grad_squared = np.square(gradients).sum()
        lr = self.lr * self.lr_increase
        while (not self.ajimo_goldstein_condition(l2_grad_squared, gradients, lr)):
            lr = lr * self.lr_decay
        self.lr = lr

    def update_thetas(self):
        self.old_thetas = self.thetas
        gradients = self.compute_gradients()
        self.ajimo(gradients)
        self.thetas = self.thetas - (self.lr * gradients)

    
    def update_costs(self):
        tmpold = self.oldcost
        self.oldcost = self.newcost
        self.newcost = self.mean_squared_error()
        return tmpold


    def undo_update_costs(self, tmpold):
        self.newcost =self.oldcost
        self.oldcost = tmpold


    def training_loop(self):
        self.start_time = datetime.now()
        self.newcost = self.mean_squared_error()
        while (not self.time_stop()):
            tmpold = self.oldcost
            self.update_thetas()
            tmpold = self.update_costs()
            if (self.newcost > self.oldcost):
                print("lol")
                self.lr = self.lr * self.lr_decay
                self.thetas = self.old_thetas
                self.undo_update_costs(tmpold)


    def middle_error(self):
        middle_thetas = (self.thetas + self.old_thetas) / 2
        return self.mean_squared_error(middle_thetas)


    def unstandardize_thetas(self):
        self.thetas[1:] = self.thetas[1:] / self.dataset.x_scaler.scale_
        self.thetas[0] = self.thetas[0] - np.dot(self.thetas[1:], self.dataset.x_scaler.mean_)
        self.thetas = self.thetas * self.dataset.y_scaler.scale_
        self.thetas[0] += self.dataset.y_scaler.mean_

    def write_thetas_to_file(self, filename="thetas.csv"):
        if self.standardize:
            self.unstandardize_thetas()
        with open(filename, "w+") as file:
            file.write(",".join([str(x) for x in self.thetas]))


    def __str__(self):
        return (f"Cost: {self.newcost}, Thetas: {self.thetas}, LR {self.lr:4.2E}")
