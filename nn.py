import numpy as np
from scipy.optimize import minimize, check_grad
import random
from splitData import getData

data = getData()
X = data[0]
y = data[1]
Xt = data[2]
yt = data[3]

print yt

def sigmoid(a):
    b = []
    for i in range(0,len(a)):
        b.append(1/(1+np.exp(-a[i])))
    return np.array(b)

def sigmoidGradient(a):
    return sigmoid(a)*(1-sigmoid(a))

class Network:
    def __init__(self,architecture):
        # to start with i'm going to only code for a simple three layer network
        self.architecture = architecture
        
        w1 = [] # weights from input to hidden layer
        for i in range(0,architecture[1]*(architecture[0]+1)):
            w1.append(random.random())
        self.w1 = np.array(w1)

        w2 = [] # weights from hidden to output layer
        for i in range(0,architecture[2]*(architecture[1]+1)):
            w2.append(random.random())
        self.w2 = np.array(w2)

    def setParameters(self,theta):
        self.w1 = np.array(theta[:self.architecture[1]*(self.architecture[0]+1)])
        self.w2 = np.array(theta[self.architecture[1]*(self.architecture[0]+1):])
        
    def predict(self,X):
        m = len(X)
        y = []

        Theta1 = self.w1.reshape(self.architecture[1],self.architecture[0]+1)
        Theta2 = self.w2.reshape(self.architecture[2],self.architecture[1]+1)

        print Theta1

        for i in range(0,m):
            x = np.hstack((np.array([1.0]),np.array(X[i])))
            h1 = sigmoid(np.matmul(Theta1,x))
            h2 = sigmoid(np.matmul(Theta2,np.hstack((np.array([1]),h1))))

            best = 0
            for j in range(0,len(h2)):
                if h2[j] > best:
                    prediction = j
                    best = h2[j]

            y.append(j)

        Theta1.reshape(self.architecture[1]*(self.architecture[0]+1),1)
        Theta2.reshape(self.architecture[2]*(self.architecture[1]+1),1)

        return y
                       
    def computeCost(self,X,y):
        m = len(X)
        J = 0.0

        Theta1 = self.w1.reshape(self.architecture[1],self.architecture[0]+1)
        Theta2 = self.w2.reshape(self.architecture[2],self.architecture[1]+1)

        Theta1_grad = np.zeros((self.architecture[1],self.architecture[0]+1))
        Theta2_grad = np.zeros((self.architecture[2],self.architecture[1]+1))

        for i in range(0,m):

            a1 = np.hstack((np.array([1.0]),np.array(X[i])))
            
            z2 = np.matmul(Theta1,a1)

            a2 = np.hstack((np.array([1.0]),sigmoid(z2)))
            h = sigmoid(np.matmul(Theta2,a2))

            y_ = [0.0]*self.architecture[-1]
            y_[y[i]] = 1.0
            y_ = np.array(y_)

            J += -np.log(h).dot(y_)-np.log(1-h).dot(1-y_)

            d3 = h-y_
            d2 = np.matmul(d3,Theta2)

            # CHANGED THE BELOW FROM MATRIX TO SCALALR MULTIPLICATION ON WHIM
            d2 = d2*sigmoidGradient(np.hstack((np.array([0]),z2)))
            d2 = np.array(d2[1:])

            # ALL OF THE ORDER HERE NEEDS CHECKING
            Theta1_grad += np.dot(d2[:,None],a1[None,:])
            Theta2_grad += np.matmul(d3[:,None],a2[None,:])

        J = J/m
        Theta1_grad = Theta1_grad/m
        Theta2_grad = Theta2_grad/m

        Theta1.reshape(self.architecture[1]*(self.architecture[0]+1),1)
        Theta2.reshape(self.architecture[2]*(self.architecture[1]+1),1)

        grad = np.vstack([Theta1_grad.reshape(self.architecture[1]*\
                                             (self.architecture[0]+1),1),
                         Theta2_grad.reshape(self.architecture[2]*\
                                             (self.architecture[1]+1),1)])

        return J, grad

    def obj_fun(self,x):
        
        J = 0.0
        
        Theta1 = np.array(x[:self.architecture[1]*(self.architecture[0]+1)])
        Theta1 = Theta1.reshape(self.architecture[1],self.architecture[0]+1)

        Theta2 = np.array(x[self.architecture[1]*(self.architecture[0]+1):])
        Theta2 = Theta2.reshape(self.architecture[2],self.architecture[1]+1)

        for i in range(0,len(self.yt)):

            a1 = np.hstack((np.array([1.0]),np.array(self.Xt[i])))
            
            z2 = np.matmul(Theta1,a1)

            a2 = np.hstack((np.array([1.0]),sigmoid(z2)))
            h = sigmoid(np.matmul(Theta2,a2))

            y_ = [0.0]*self.architecture[-1]
            y_[self.yt[i]] = 1.0
            y_ = np.array(y_)

            J += -np.log(h).dot(y_)-np.log(1-h).dot(1-y_)

        return J/len(self.yt)

    def grads(self,x):

        m = len(self.yt)

        Theta1 = np.array(x[:self.architecture[1]*(self.architecture[0]+1)])
        Theta1 = Theta1.reshape(self.architecture[1],self.architecture[0]+1)

        Theta2 = np.array(x[self.architecture[1]*(self.architecture[0]+1):])
        Theta2 = Theta2.reshape(self.architecture[2],self.architecture[1]+1)

        Theta1_grad = np.zeros((self.architecture[1],self.architecture[0]+1))
        Theta2_grad = np.zeros((self.architecture[2],self.architecture[1]+1))

        for i in range(0,m):

            a1 = np.hstack((np.array([1.0]),np.array(self.Xt[i])))           
            z2 = np.matmul(Theta1,a1)

            a2 = np.hstack((np.array([1.0]),sigmoid(z2)))
            h = sigmoid(np.matmul(Theta2,a2))

            y_ = [0.0]*self.architecture[-1]
            y_[self.yt[i]] = 1.0
            y_ = np.array(y_)

            d3 = h-y_
            d2 = np.matmul(d3,Theta2)

            # CHANGED THE BELOW FROM MATRIX TO SCALALR MULTIPLICATION ON WHIM
            d2 = d2*sigmoidGradient(np.hstack((np.array([0]),z2)))
            d2 = np.array(d2[1:])

            # ALL OF THE ORDER HERE NEEDS CHECKING
            Theta1_grad += np.dot(d2[:,None],a1[None,:])
            Theta2_grad += np.matmul(d3[:,None],a2[None,:])

        Theta1_grad = Theta1_grad/m
        Theta2_grad = Theta2_grad/m


        Theta1_grad = Theta1_grad.reshape(1,self.architecture[1]*(self.architecture[0]+1))
        Theta2_grad = Theta2_grad.reshape(1,self.architecture[2]*(self.architecture[1]+1))
        
        grad = []
        
        for i in range(0,len(Theta1_grad[0])):
            grad.append(Theta1_grad[0][i])

        for i in range(0,len(Theta2_grad[0])):
            grad.append(Theta2_grad[0][i])
            
#        grad = np.hstack([Theta1_grad.reshape(1,self.architecture[1]*\
#                                             (self.architecture[0]+1)),
#                         Theta2_grad.reshape(1,self.architecture[2]*\
#                                             (self.architecture[1]+1))])
#
        return np.array(grad)
    
    def train(self,Xt,yt):

        self.Xt = Xt
        self.yt = yt

        x0 = []
        for i in range(0,self.architecture[1]*(self.architecture[0]+1)+
                       test.architecture[2]*(test.architecture[1]+1)):
            x0.append(random.random())

        x0 = np.array(x0)

        print 'Checking the gradients... ',
        print check_grad(self.obj_fun, self.grads, x0)

        res = minimize(self.obj_fun, x0, method='BFGS', jac=self.grads)

        self.w1 = res.x[:self.architecture[1]*(self.architecture[0]+1)]
        self.w2 = res.x[self.architecture[1]*(self.architecture[0]+1):]

        return ''

test = Network([2,12,5])
print test.w1
print test.predict([[1.5,0.2],[0.8,1.0]])
test.train([[1.5,0.2],[0.8,1.0]],[3,2])
print test.w1

print test.predict([[1.5,0.2],[0.8,1.0]])
