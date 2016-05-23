__author__ = 'takacs'
from matplotlib.pyplot import scatter, title, xlabel, ylabel, show, plot
from numpy import loadtxt, zeros, ones


def drowplot():
    # load the dataset
    data = loadtxt('ex1data1.txt', delimiter=',')
    # plot the data
    scatter(data[:, 0], data[:, 1], marker='o', c='b')
    title('Profits distribution')
    xlabel('Population of city 10,000s')
    ylabel('Profit in $10,000s')
    show()


def drowplot1():
    # load the dataset
    data = loadtxt('ex1data1.txt', delimiter=',')
    # plot the data
    X = data[:, 1];
    Y = data[:, 2]
    m = len(Y)
    plot(X, Y, 'rx', 'MarkerSize', 10)
    title('Profits distribution')
    ylabel('Profit in $10,000s')
    xlabel('Population of City in 10,000s')
    show()


def computecost(x, y, theta):
    '''
     Comput cost for linear regression
    '''

    # Number of training samples
    m = y.size
    predictions = x.dot(theta).flatten()
    # print('X' + repr(x))
    # print('Predictions' + repr(predictions))
    sqErrors = (predictions - y) ** 2
    # print('Y' + repr(y))
    # print('predictions-Y' + repr(predictions-y))
    # print('sqErrors' + repr(sqErrors))

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(x, y, theta, alpha, iterations):
    '''
    Performs gradient descent to learn theta
    by taking iterations gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros((iterations, 1))

    for i in range(iterations):
        predictions = x.dot(theta).flatten()

        errors_x1 = (predictions - Y) * x[:, 0]
        # print(errors_x1)
        errors_x2 = (predictions - Y) * x[:, 1]
        # print(errors_x2)

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i][0] = computecost(x, y, theta)

    return 'Theta::' + repr(theta), 'J_history::' + repr(J_history)


if __name__ == "__main__":
    print('Maschine Learning course examples....')
    # drowplot1()
    # load the dataset
    data = loadtxt('ex1data1.txt', delimiter=',')
    Y = data[:, 1]
    X = data[:, 0]
    # Number of training samples
    m = Y.size
    # Add a column of ones to X (interception data)
    it = ones(shape=(m, 2))
    myit = ones((m, 2))
    myit[:, 1] = X
    # it[:, 1] = X

    # Initialize fitting parameters
    theta = zeros((2, 1))
    # theta = ((-3,1.2))


    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    print('Debug values....')
    # print ('The value of Y[0] is ' + repr(Y[0]))
    # print ('The value of Y[0] is  %5.5f' %Y[0])
    # print ('The number of training samples %6d' %m)
    # print ('Population of cities ' + repr(X[0]))
    # print ('Print it ' + repr(it))
    # print ('Print myit ' + repr(myit))
    # print ('Theta ' + repr(theta))

    # compute and display initial cost
    # print (computecost(myit, Y, theta))
    print(gradient_descent(myit, Y, theta, alpha, iterations))
