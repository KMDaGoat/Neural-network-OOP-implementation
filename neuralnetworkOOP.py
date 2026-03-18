import numpy as np
from sklearn.datasets import make_moons

x , y = make_moons(n_samples = 500 , noise = 0.2)
x = x.T
y = y.reshape(1,-1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def inputnormalisation(x):
    mean = np.mean(x , axis = 1 , keepdims = True)
    stdev = np.std(x , axis = 1 , keepdims = True)
    xnorm = (x - mean) /stdev
    return xnorm

xnorm = inputnormalisation(x)

class neuralnetwork:
    def __init__(self , xnorm , y , lambd , epsilon):
        self.layers = [2,5,4,1]
        self.x = xnorm
        self.y = y
        self.lambd = lambd
        self.epsilon = epsilon
        self.thetaparam =[]
        self.keysparam = []
        self.thetagrad = []
        self.keysgrad = []

    def intialisation(self):
        self.layernum = len(self.layers)
        self.parameters = {}
        np.random.seed(3)

        for layer in range(1 , self.layernum):
            self.parameters["W"+ str(layer)] = (np.random.randn(self.layers[layer] , self.layers[layer-1])) * np.sqrt(1 / self.layers[layer-1])
            self.parameters["b" + str(layer)] = np.zeros((self.layers[layer] ,1))

        return self.parameters

    def forwardprop(self, parameterbeingused , gradientcheck = False):
        if gradientcheck:
            self.parameters = parameterbeingused

        self.cache = []
        self.layernum = len(self.parameters) // 2
        A = self.x
        for layer in range(1 , self.layernum):
            w = self.parameters["W" + str(layer)]
            b = self.parameters["b" + str(layer)]
            Aprev = A
            z = np.dot(w , Aprev) + b
            A = np.tanh(z)
            cache = (Aprev , w, b , z)
            self.cache.append(cache)

        w = self.parameters["W" + str(self.layernum)]
        b = self.parameters["b" + str(self.layernum)]
        Aprev = A
        z = np.dot(w , A) + b
        self.output = sigmoid(z)
        cache = (Aprev , w , b , z)
        self.cache.append(cache)

        return self.cache , self.output

    def costfunction(self):
        m = self.y.shape[1]
        self.cost = (-1 / m) * np.sum(self.y * np.log(self.output) + (1 - self.y) * np.log(1 - self.output))

        l2cost = 0
        for key in self.parameters:
            if key.startswith("W"):
                l2cost += np.sum(np.square(self.parameters[key]))

        l2cost = (self.lambd / (2 * m)) * l2cost

        self.cost += l2cost
        return np.squeeze(self.cost)

    def backwardprop(self):
        m = self.x.shape[1]
        self.layernum = len(self.cache)
        self.grads = {}

        Aprev , w , b , z = self.cache[self.layernum - 1]
        dz = self.output - self.y
        self.grads["dw" + str(self.layernum)] = (1/m) * np.dot(dz , Aprev.T)
        self.grads["db" + str(self.layernum)] = (1/m) * np.sum(dz , axis =1 , keepdims = True)
        dA = np.dot(w.T , dz)

        for layer in range(self.layernum-1 , 0 , -1):
            Aprev , w ,b ,z = self.cache[layer-1]
            dz = dA * (1 - np.tanh(z)**2)
            self.grads["dw" + str(layer)] = (1 / m) * np.dot(dz, Aprev.T)
            self.grads["db" + str(layer)] = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dA = np.dot(w.T , dz)

        return self.grads

    def updateparameters(self , learningrate):
        self.layernum = len(self.parameters) // 2
        for layer in range(1 , self.layernum + 1):
            self.parameters["W" + str(layer)] -= self.grads["dw" + str(layer)] * learningrate
            self.parameters["b" + str(layer)] -= self.grads["db" + str(layer)] * learningrate

        return self.parameters

    def vectortodict(self , currentvector):
        length = len(self.parameters) // 2
        returnedparameter = {}
        pointer = 0
        for layer in range(1 , length + 1):
            for values in ["W" , "b"]:
                key = values + str(layer)
                shape = self.parameters[key].shape
                numofelements = np.prod(shape)
                returnedparameter[key] = currentvector[pointer : pointer + numofelements].reshape(shape)
                pointer += numofelements

        return returnedparameter

    def gradientchecking(self):
        self.layernum = len(self.parameters)//2
        tempparam = self.parameters.copy()
        tempgrads = self.grads.copy()

        #turning the parameter dictionary into a vector 
        for layer in range( 1, self.layernum + 1):
            for values in ["W" , "b"]:
                key = values + str(layer)
                vector = tempparam[key].reshape(-1)
                self.thetaparam.append(vector)
                self.keysparam.extend([key] * vector.shape[0])

        self.thetaparam = np.concatenate(self.thetaparam)

        #turning the grads dictionary into a vector 
        for layer in range(1 , self.layernum + 1):
            for values in ["dw" , "db"]:
                key = values + str(layer)
                vector = tempgrads[key].reshape(-1)
                self.thetagrad.append(vector)
                self.keysgrad.extend([key] * vector.shape[0])

        self.thetagrad = np.concatenate(self.thetagrad)

        #making a vector to hold all the cosrts returned later on
        self.approxgrad = np.zeros_like(self.thetaparam)

        #gradient checking
        for i in range(len(self.thetaparam)):
            thetaplus = self.thetaparam.copy()
            thetaminus = self.thetaparam.copy()

            thetaplus[i] += self.epsilon
            thetaminus[i] -= self.epsilon

            self.plusdict = self.vectortodict(thetaplus)
            self.minusdict = self.vectortodict(thetaminus)

            _, plusforward = self.forwardprop(self.plusdict, True)
            pluscost = self.costfunction()

            _, minusforward = self.forwardprop(self.minusdict, True)
            minuscost = self.costfunction()

            self.approxgrad[i] = (pluscost - minuscost) / (2 * self.epsilon)

        numerator = np.linalg.norm(self.thetagrad - self.approxgrad)
        denominator = np.linalg.norm(self.thetagrad) + np.linalg.norm(self.approxgrad)
        difference = numerator / denominator

        return difference

    def nnrun(self , iterations , learningrate):
        gradientchecked = False
        for iteration in range(iterations):
            self.forwardprop(None , False)
            self.costfunction()
            self.backwardprop()
            if not gradientchecked :
                self.gradientchecking()
                self.forwardprop(None , False)
                self.costfunction()
                self.backwardprop()
                gradientchecked = True
            self.updateparameters(learningrate)

            if iteration % 1000 == 0:
                print(f"cost at {iteration} : {self.cost}")

        predictions = (self.output > 0.5).astype(int)
        return predictions

neuralnetworkone = neuralnetwork(xnorm , y , 0.01 , 1e-7)
neuralnetworkone.intialisation()

predictions = neuralnetworkone.nnrun(10001 , 0.1)
accuracy = np.mean(predictions == y)
print(f"accuracy: {accuracy * 100:.2f}%")
