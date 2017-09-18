"""Script to do some basic machine learning"""
import sys

def main():
    """Main entry point for the script."""
    naive_bayes()
    pass

def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit([[1, 2], [3, 4], [-1, -2], [-3, -4]], [1, 1, 2, 2])
    GaussianNB(priors=None)
    print(clf.predict([[-1, -1]]))

if __name__ == '__main__':
    sys.exit(main())