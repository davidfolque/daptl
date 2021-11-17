#imports
import numpy
import matplotlib.pyplot as plt
from datetime import date
plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (8,6)
#plt.tight_layout()


####################
#functions
def sigma(x):
    return 1/(1 + numpy.exp(-x))

def model(w, theta, x, M):
    #outputs the probability of class 1!
    layer1 = (sigma(M * w) * theta) @ x
    p = sigma(1 * layer1)
    return p

def gModel(w, theta, x, M, m):
    mHard = sigma(M * w)
    mSoft = sigma(m * w)
    layer1 = (mHard * theta) @ x
    p = sigma(1 * layer1)
    gW = p * (1 - p) * (m * mSoft * (1 - mSoft)) * theta * x
    gTheta = p * (1 - p) * mHard * x
    #gTheta = p * (1 - p) * mSoft * x
    return gW, gTheta

def logisticLoss(p1, y):
    return (1 - y) * numpy.log(1 - p1) + y * numpy.log(p1)

def gLogisticLoss(p1, y):
    return - (1 - y)/(1 - p1) + y / p1

#objective and optimization
def objective(w, theta, X, parameters):
    #images = [train, val, test]
    #train = [[image0, image1], [image0, image1]...] 
    ell, m, M, eta, T = parameters
    obj = 0
    for n in range(len(X)):
        for i in range(len(X[n])):
            x = X[n][i]
            y = i
            out = model(w, theta, x, M)
            obj = obj + logisticLoss(out, y)
    #add sparsity term
    obj = obj/(2 * len(X)) + ell * sum(1 * (w > 0))
    return obj 

def gObjective(w, theta, X, parameters):
    #X[batchLength][2]
    ell, m, M, eta, T = parameters
    obj = 0
    grad = [numpy.zeros(len(w)), numpy.zeros(len(theta))]
    for n in range(len(X)):
        for i in range(len(X[n])):
            x = X[n][i]
            y = i
            out = model(w, theta, x, M)
            loss = logisticLoss(out, y)
            gLoss = gLogisticLoss(out, y)
            gW, gTheta = gModel(w, theta, x, M, m)
            grad[0] = grad[0] + gLoss * gW
            grad[1] = grad[1] + gLoss * gTheta
    grad = [x/(2 * len(X)) for x in grad]
    return grad

def gradientStep(w, theta, X, parameters):
    ell, m, M, eta, T = parameters
    gW, gTheta = gObjective(w, theta, X, parameters)
    gW = gW + ell * 2 * (numpy.ones(len(w)) + w)
    gTheta = gTheta + ell * 2 * theta
    w = w - eta * gW
    theta = theta - eta * gTheta
    return w, theta

def trainModel(w, theta, data, parameters):
    sparsity = .1
    dataTrain, dataTest = data
    ell, m, M, eta, T = parameters
    obj = [objective(w, theta, dataTrain, parameters)]
    stopIt = 0
    batchSize = 10#len(Y)
    t = 0
    while stopIt == 0:
        batch = numpy.random.choice(len(dataTrain), batchSize, replace=False)
        X = [dataTrain[i] for  i in batch]
        wOld = w
        thetaOld = theta
        w, theta = gradientStep(w, theta, X, parameters)
        t = t + 1
        if (t % 500) == 0: 
            loss = objective(w, theta, dataTrain, parameters)
            obj.append(loss)
            print(len(obj), obj[-1], sum(w>0)) 
        if t > T or eta < 1e-6:
            stopIt = 1
        """
        if sum(w>0) <= len(w) * (1 - sparsity):
            w = wOld
            #stopIt = 1
        if obj[-1] >= obj[-2]:
            obj[-1] = obj[-2]
            #eta = eta/10
            w = wOld
            theta = thetaOld
            #print(eta)
            parameters = ell, m, M, eta, T
        """
    a = [w, theta]
    return a, obj

def getRandomImages(data, classes):
    nMax = min([len(data[i]) for i in classes])
    images = []
    for n in range(nMax):
        pair = [data[classes[0]][n], data[classes[1]][n]]
        images.append(pair)
    return images

###################################
figurePath = "."
# dd/mm/YY
today = date.today()
date = today.strftime("%d/%m/%Y")
dataPath = "/Users/ugqm002/Documents/data/mnist/"
localDataPath = "./mnist/temp/"
date = date + 'mnist'
longNames = [
        'One', 
        'Two', 
        'Three', 
        'Four', 
        'Five',
        'Size', 
        'Seven', 
        'Eight', 
        'Nine'
        ]
shortNames = [str(i) for i in range(1, 10)]
classNames = shortNames
    
#load data set
"""
allSplitDataReducedName = localDataPath + "allSplitDataReduced.npy"
print('loading split small dataset from ' 
            + allSplitDataReducedName +  '...')
fullTrain, fullTest = numpy.load(allSplitDataReducedName, allow_pickle = True)
"""

allSplitDataName = localDataPath + "allSplitData.npy"
print('loading split large dataset from ' 
            + allSplitDataName +  '...')
fullTrain, fullTest = numpy.load(allSplitDataName, allow_pickle = True)

fullData = fullTrain, fullTest
D = 64
print('D=' + str(D))
print(len(fullTrain[0][1]))

print(sum([len(t) for t in fullTrain]))
plt.imshow(numpy.array(fullTrain[9][0]).reshape(8,8))
plt.show()
exit()


numpy.random.seed(12334)

classes = 0, 1
selectedData = []
for data in fullData:
    images = getRandomImages(data, classes)
    selectedData.append(images)
print([len(x) for x in selectedData])
print([len(x) for x in selectedData[1][:5]])

w0 = 1 * abs(numpy.random.randn(D))
theta0 = 1 * numpy.random.randn(D)
ell = .02
M = 1000
eta = .05
T = 10000


allm = []
modelSize = []

numpy.random.seed(12345678)
m = M
allm.append('t_s=' + str(m) + ', ')
parameters = ell, m, M, eta, T
a, obj = trainModel(w0, theta0, selectedData, parameters)
plt.plot(obj)
w, theta = a

q = theta * sigma(M * w)
active = sum(1 *(w > 0)) + 3
print(numpy.sort(abs(q))[-active:])
print(numpy.argsort(abs(q))[-active:])

modelSize.append('|m|=' + str(sum(1 *(w > 0))))

numpy.random.seed(12345678)
m = M / 100
allm.append('t_s=' + str(m) + ', ')
parameters = ell, m, M, eta, T
a, obj = trainModel(w0, theta0, selectedData, parameters)
plt.plot(obj)
w, theta = a

q = theta * sigma(M * w)
active = sum(1 *(w > 0)) + 3
print(numpy.sort(abs(q))[-active:])
print(numpy.argsort(abs(q))[-active:])

modelSize.append('|m|=' + str(sum(1 *(w > 0))))

numpy.random.seed(12345678)
m = M / 1000
allm.append('t_s=' + str(m) + ', ')
parameters = ell, m, M, eta, T
a, obj = trainModel(w0, theta0, selectedData, parameters)
plt.plot(obj)
w, theta = a

q = theta * sigma(M * w)
active = sum(1 *(w > 0)) + 3
print(numpy.sort(abs(q))[-active:])
print(numpy.argsort(abs(q))[-active:])

modelSize.append('|m|=' + str(sum(1 *(w > 0))))

plt.legend([allm[i] + modelSize[i] for i in range(3)])
plt.xlabel('sgd iterations')
plt.ylabel('objective value')
plt.title('t_l=' + str(M))
plt.show()




