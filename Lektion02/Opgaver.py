import numbers
import sys

import matplotlib.pyplot as plt
import numpy as np

# Opgave 2
X = 2 * np.random.randn(100,1)
y = 4 + 3 * X + np.random.randn(100,1)

plt.plot(X, y, "b.")
plt.show()

#Opgave 3 - beregn cost function
#a theta0, b er theta1
def cost(a, b, X, y):
    m = len(y)
    error = a + b * X - y
    J = np.sum(error ** 2)/(2*m)
    return J

ainterval = np.arange(0.1,10,0.1)
binterval = np.arange(0.1,10,0.1)

# Lists for finding smallest cost
costs = []
a_list = []
b_list = []
for atheta in ainterval:
    for btheta in binterval:
        a_list.append(atheta)
        b_list.append(btheta)
        costs.append(cost(atheta, btheta, X, y)) #make list to find minimum later
    print("xy %f.%f.%f" % (atheta, btheta, cost(atheta, btheta, X, y)))
        #

#Extra funktion til at finde mindste cost
def smallest_cost(cost_list : [], a_list : [], b_list : []):
    temp = sys.maxsize
    index = 0
    for id, cost in enumerate(cost_list):
        if(cost < temp):
            temp = cost
            index = id

    return "Lowest cost: %f. Best atheta: %f. Best btheta %f." % (temp, a_list[index], b_list[index])

print(smallest_cost(costs, a_list, b_list))

#Opgave 4
from mpl_toolkits.mplot3d import Axes3D


def cost(a, b):
    ### Evaluate half MSE (Mean square error)
    m = len(Ydots)
    error = a + b * Xdots - Ydots
    J = np.sum(error ** 2) / (2 * m)
    return J


Xdots = 2 * np.random.rand(100, 1)
Ydots = -5 + 7 * Xdots + np.random.randn(100, 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ainterval = np.arange(-10, 10, 0.05)
binterval = np.arange(-10, 10, 0.05)

X, Y = np.meshgrid(ainterval, binterval)
zs = np.array([cost(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)
ax.set_xlabel('Thete0')
ax.set_ylabel('Theta1')
ax.set_zlabel('Cost')
plt.show()

#Opgave 5
def linear_regression(X, y, theta0=0, theta1=0, epochs=10000, learning_rate=0.0001):
    N = float(len(y))
    for i in range(epochs):
        y_hypothesis = (theta1 * X) + theta0
        cost = sum([data ** 2 for data in (y - y_hypothesis)]) / N
        theta1_gradient = -(2 / N) * sum(X * (y - y_hypothesis))
        theta0_gradient = -(2 / N) * sum(y - y_hypothesis)
        theta0 = theta0 - (learning_rate * theta0_gradient)
        theta1 = theta1 - (learning_rate * theta1_gradient)

    return theta0, theta1, cost


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

theta0, theta1, cost = linear_regression(X, y, 0, 0, 1000, 0.01)

plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])

# lets plot that line:
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot([theta0, theta1])
plt.plot(X_new, y_predict, "g-")

plt.show()

#opgave 7
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X * X + X + 2 + np.random.randn(100, 1)
plt.plot(X,y, "g.")
plt.axis([-3,3,0,10])
poly_features = PolynomialFeatures(2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lm = LinearRegression()
lm.fit(X_poly, y)
#fit function
# lm.coef er hældningskoefficient; altså når man går en hen af x, hvor meget går skal man så op af y
# lm.intercept er hvor den skære y-aksen
f = lambda x: lm.coef_[0][1]*x*x + lm.coef_[0][0]*x + lm.intercept_
plt.plot(X,f(X), "b.")
plt.show()