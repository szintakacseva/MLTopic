__author__ = 'takacs'
from sklearn import datasets

if __name__ == "__main__":
    print('Scikit-learn tutorial examples....')
    digits = datasets.load_digits()
    iris = datasets.load_iris()
    print('Digits data, target....')
    print(digits.data), print(digits.target)
    print('Iris data, target....')
    print(iris.data), print(iris.target)
