import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Opgave 1
# Array of points with a classification
#X = np.array(np.matrix('6,350; 2.5, 400;4.5,500; 3.5,350; 2, 300;4, 600;7, 300;5, 500;5, 400;6, 400;3, 400;4, 500;1, 200;3, 400;7, 700;3, 550;2.5, 650'))
#y = np.array(np.matrix('0;0;1;0;0;1;1;1;0;1;0;0;0;0;1;1;0'))[:, 0]

X = np.array(np.matrix('4,450;5,600;6,700;4.5,550;4.9,500;5,650;5.5,500; 5.25,525; 4.25,625; 4.75,575'))
y = np.array(np.matrix('0;1;1;0;0;1;0;1;1;1'))[:,0]

pos = np.where(y == 1)
neg = np.where(y == 0)

plt.plot(X[pos[0], 0], X[pos[0], 1], 'ro')
plt.plot(X[neg[0], 0], X[neg[0], 1], 'bo')
plt.xlim([min(X[:, 0]), max(X[:, 0])])
plt.ylim([min(X[:, 1]), max(X[:, 1])])
plt.plot()

logreg = linear_model.LogisticRegression(C=1000) #higher C risks overfitting
model = logreg.fit(X, y)
#model.coef_[0,0]*x + model.coef_[0,1]*y + model.intercept_[0] = 0
#y = - ( model.coef_[0,0]*x +  model.intercept_[0]) / model.coef_[0,1]
xx = np.linspace(1, 7)
yy = -  (model.coef_[0,0] / model.coef_[0,1]) * xx - ( model.intercept_[0]  /  model.coef_[0,1])
plt.plot(xx, yy, 'k-')

plt.show()


#Opgave 2

X = np.array(np.matrix('34.62365962451697,78.0246928153624; 30.28671076822607,43.89499752400101;'
                       ' 35.84740876993872,72.90219802708364;60.18259938620976,86.30855209546826;'
                       '79.0327360507101,75.3443764369103; 45.08327747668339,56.3163717815305; '
                       '61.10666453684766,96.51142588489624; 75.02474556738889,46.55401354116538; '
                       '76.09878670226257,87.42056971926803;84.43281996120035,43.53339331072109; '
                       '95.86155507093572,38.22527805795094; 75.01365838958247,30.60326323428011; '
                       '82.30705337399482,76.48196330235604; 69.36458875970939,97.71869196188608; '
                       '39.53833914367223,76.03681085115882; 53.9710521485623,89.20735013750205;'
                       ' 69.07014406283025,52.74046973016765; 67.94685547711617,46.67857410673128; '
                       '70.66150955499435,92.92713789364831; 76.97878372747498,47.57596364975532;'
                       '67.37202754570876,42.83843832029179'))
y = np.array(np.matrix('0;0;0;1;1;0;1;1;1;1;0;0;1;1;0;1;1;0;1;1;0'))[:,0]

pos = np.where(y == 1)
neg = np.where(y == 0)

plt.plot(X[pos[0], 0], X[pos[0], 1], 'ro')
plt.plot(X[neg[0], 0], X[neg[0], 1], 'bo')

plt.xlim([min(X[:, 0]), max(X[:, 0])])
plt.ylim([min(X[:, 1]), max(X[:, 1])])

logreg = linear_model.LogisticRegression(C = 1000)
model = logreg.fit(X, y)

xx = np.linspace(0, 100) #data sæt går fra 0 til 100 så vi vil tegne linjen i det space
yy = -  (model.coef_[0,0] / model.coef_[0,1]) * xx - ( model.intercept_[0]  /  model.coef_[0,1])
plt.plot(xx, yy, 'k-')

plt.show()

X = np.array(np.matrix('10000,50000, 0; 5000, 1000, 1; 20000, 10000, 1; 300, 1000, 0; 5000, 300, 1; 40000, 10000, 1; 3000, 5000, 0; 10000, 20000, 0; 50000, 60000, 0; 40000, 60000, 0; 90000, 95000, 0; 50000, 45000, 1'))
y = np.array(np.matrix('0;1;1;0;1;1;0;0;0;0;0;1'))[:,0]

pos = np.where(y == 1)
neg = np.where(y == 0)

plt.plot(X[pos[0], 0], X[pos[0], 1], 'ro')
plt.plot(X[neg[0], 0], X[neg[0], 1], 'bo')

plt.xlim([min(X[:, 0]) - 2000, max(X[:, 0]) + 2000])
plt.ylim([min(X[:, 1]) - 2000, max(X[:, 1]) + 2000])

logreg = linear_model.LogisticRegression(C = 1000)
model = logreg.fit(X, y)

xx = np.linspace(-10000, 100000) #data sæt går fra 0 til 100 så vi vil tegne linjen i det space
yy = -  (model.coef_[0,0] / model.coef_[0,1]) * xx - ( model.intercept_[0]  /  model.coef_[0,1])
plt.plot(xx, yy, 'k-')

plt.show()

#opgave 3
# Array of points with a classification
X = np.array(np.matrix('2,300;4,600;7,300;5,500;5,400;6,400;3,400;4,500;1,200;3,400;7,700;3,550;2.5,650'))
y = np.array(np.matrix('0;1;1;1;0;1;0;0;0;0;1;1;0'))[:, 0]

pos = np.where(y == 1)
neg = np.where(y == 0)

plt.plot(X[pos[0], 0], X[pos[0], 1], 'ro')
plt.plot(X[neg[0], 0], X[neg[0], 1], 'bo')
plt.xlim([min(X[:, 0]), max(X[:, 0])])
plt.ylim([min(X[:, 1]), max(X[:, 1])])

#make a classifier
logreg = linear_model.LogisticRegression(C=5) #C started at 10000

h = .02  # step size in the mesh

# we fit the data.
logreg.fit(X, y)

score = logreg.score(X, y)

print('score: ' + str(score))
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Ravel() returns a contiguous flattedned array
# Predict the color
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot. Reshape point to xx format
Z = Z.reshape(xx.shape)

# figure(Num = our figure number, figsize=(1,1)) creates an inch-by-inch image
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

#Show plot
plt.show()