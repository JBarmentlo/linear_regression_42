float_formatter = "{:.2E}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def normalize(a):
    col_sums = a.sum(axis=0)
    new_matrix = a / col_sums[np.newaxis, :]
    return new_matrix

class Shaman():
    def __init__(self, dataset):
        self.p = dataset.p
        self.dataset = dataset
        self.thetas = np.zeros([self.p + 1], dtype = float)
        self.old_thetas = self.thetas
        self.lr = 1.0
        self.lr_decay = 1.0 / 2
        self.mininal_improvement = 0.1
        self.newcost = 0.0
        self.oldcost = 0.0


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
        gradients = np.dot(error, normalize(self.dataset.x))
        gradients = gradients / len(self.dataset.y)
        return (gradients)


    def update_thetas(self):
        self.old_thetas = self.thetas
        self.thetas = self.thetas - (self.lr * self.compute_gradients())

    
    def update_costs(self):
        tmpold = self.oldcost
        self.oldcost = self.newcost
        self.newcost = self.mean_squared_error()
        return tmpold


    def undo_update_costs(self, tmpold):
        self.newcost =self.oldcost
        self.oldcost = tmpold


    def training_loop(self):
        keep_learning = True
        self.newcost = self.mean_squared_error()
        while (keep_learning):
            tmpold = self.oldcost
            self.update_thetas()
            tmpold = self.update_costs()
            if (self.newcost > self.oldcost):
                self.lr = self.lr * self.lr_decay
                self.thetas = self.old_thetas
                self.undo_update_costs(tmpold)
            keep_learning = not self.should_i_stop()
            print(self)


    def middle_error(self):
        middle_thetas = (self.thetas + self.old_thetas) / 2
        return self.mean_squared_error(middle_thetas)

    
    def should_i_stop(self):
        if abs(self.oldcost - self.newcost) > self.mininal_improvement:
            return False
        if abs(self.middle_error() - self.newcost) > self.mininal_improvement:
            return False
        return True



    def __str__(self):
        return (f"Cost: {self.newcost:.2e}, Thetas: {self.thetas}, LR {self.lr:4.2E}")
