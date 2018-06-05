from sklearn import preprocessing
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

class MySVM(object):
    """
    SVM Class to encapsulate all SVM functionalities to predict on given data.
    """

    """
    This is a measure of the means and standard deviations of each vector of each of the training instances.
    This is used to normalize any data on which the classifier has to predict.
    """
    std_scale = None

    """
    Main SVM classifier this class trains and uses for the actual classification.
    """
    classifier = None

    
    def __init__(self):
        """
        Constructor which initializes a SVM with a linear kernel as it is best suited for this domain.
        """
        self.classifier = SVC(kernel='linear')

    
    def compute_std_scale(self, train_data):
        """
        Method to compute the means & standard deviations of each feature from the training data
        which will be used to normalize (Z-score method) any data on which prediction is performed.
        """
        self.std_scale = preprocessing.StandardScaler().fit(train_data)

    
    def fit(self, train_data, labels):
        """
        Method to train SVM on training data.
        """
        self.compute_std_scale(train_data) #Computing standard deviations & means from training data
        x_train_std = self.std_scale.transform(train_data) #Normalizing training data using Z-score method
        self.classifier.fit(x_train_std, labels)

    def predict(self, test_data):
        """
        Method to predict on raw, un-normalized data.
        """
        x_test_std = self.std_scale.transform(test_data) #Normalizing data using the means & standard deviations from the training data
        #plot_data([test_data[0], x_test_std[0]], ['org', 'norm'])
        x_test_std = test_data
        prd = self.classifier.predict(x_test_std)
        return prd

def plot_data(data_set, labels):
  i = 0
  for data in data_set:
    idx = np.arange(data.shape[0])
    plt.plot(idx, data, label=labels[i])
    i += 1
  plt.legend()
  plt.show()