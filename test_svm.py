# train a svm for binary classification and get normal vector of the hyperplane
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# we create 40 separable points
#X, y = make_blobs(n_samples=40, centers=2, random_state=6)

X = np.array([[  6.37734541, -10.61510727],
       [  6.50072722,  -3.82403586],
       [  4.29225906,  -8.99220442],
       [  7.39169472,  -3.1266933 ],
       [  7.64306311, -10.02356892],
       [  8.68185687,  -4.53683537],
       [  5.37042238,  -2.44715237],
       [  9.24223825,  -3.88003098],
       [  5.73005848,  -4.19481136],
       [  7.9683312 ,  -3.23125265],
       [  7.37578372,  -8.7241701 ],
       [  6.95292352,  -8.22624269],
       [  8.21201164,  -1.54781358],
       [  6.85086785,  -9.92422452],
       [  5.64443032,  -8.21045789],
       [ 10.48848359,  -2.75858164],
       [  7.27059007,  -4.84225716],
       [  6.29784608, -10.53468031],
       [  9.42169269,  -2.6476988 ],
       [  8.98426675,  -4.87449712]])
y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0])


# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)


plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

print(clf.coef_[0]/clf.coef_[0][0])


# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.savefig('testSVM.png')