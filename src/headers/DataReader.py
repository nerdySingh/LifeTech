from headers.names.featureLists import *
from headers.names.referenceFileNames import *
import cPickle as pickle
import csv

class DataReader():
    """
    Class to handle reading, cleaning and parsing of data from CSV file.
    """
    
    """
    List of columns relevant for SVM training.
    """
    impCols = None

    """
    CSV Reader object
    """
    reader = None

    """
    Indexed according to the rows as in the CSV file.
    """
    labels = []

    """
    This is a list of dictionaries with each entry corresnponding to a row in the CSV file.
    The keys correspond to the columns of the CSV file.
    """
    input_data = [] 

    """
    Dictionary containing the mapping relations between every non-numeric value and
    the corresponding numeric label assigned to it for each non-numeric column.
    """
    non_numeric_maps = None

    
    def __init__(self, fileName=None):
        """
        Class constructor which initializes the list containing names of columns relevant for SVM training
        and creates a CSV file reader object.
        """
        self.impCols = mainCols[:]
        if fileName is not None:
            f = open(fileName, 'rb')
            self.reader = csv.reader(f)

    
    def readPickleData(self, input_data_file=default_input_data_filename, label_file=default_label_filename, non_numeric_maps_file=default_non_numeric_maps_filename):
        """
        Reads data from pickled files containing pre-processed data.
        """
        self.input_data = pickle.load(open(input_data_file, 'rb'))
        self.labels = pickle.load(open(label_file, 'rb'))
        self.non_numeric_maps = pickle.load(open(non_numeric_maps_file, 'rb'))

    
    def dumpPickleData(self):
        """
        Dumps cleaned data stored in the class onto HDD.
        """
        pickle.dump(self.input_data, open(default_input_data_filename, 'wb'))
        pickle.dump(self.labels, open(default_label_filename, 'wb'))

    
    def readData(self):
        """
        Reads data and parses it to a list of dictionaries where each dictionary corresponds to a row.
        """
        rowNum = 0
        header = []
        for row in self.reader:
            if rowNum == 0: #Storing names of columns in header[] list
                header[:] = row[:]
            else:
                rowData = dict(zip(header, row)) #Mapping each column entry to the column name and storing as a dictionary for easy indexing
                try:
                    if rowData['term'].split()[0] == '36': #Filtering out entries with term not equal to 36 months
                        if rowData['loan_status'].strip() == 'Fully Paid':
                            self.labels.append(1) #Labelling Fully Paid entries as '1', all others as '0'
                        else:
                            self.labels.append(0)
                        myDict = {key : rowData[key].strip().lower() for key in self.impCols}
                        self.input_data.append(myDict)
                except:
                    continue
            rowNum += 1
        self.gen_non_numeric_maps() #Generating mappings of all non-numeric features to their equivalent integer maps.

    def get_unique_label_maps(self, col_name):
        """
        Generates string to number mappings for columns with non-numerical entries.
        """
        vect = [elem[col_name] for elem in self.input_data]
        unique_vect = set(vect)
        return dict(zip(unique_vect, self.frange(len(unique_vect)))), len(unique_vect)
    

    def gen_non_numeric_maps(self):
        """
        Generates all integer mappings for all non-numeric columns in CSV dataset.
        """
        colMaps = []
        for col in non_numeric_cols:
            colMaps.append(self.get_unique_label_maps(col)[0])
        self.non_numeric_maps = dict(zip(non_numeric_cols, colMaps))

    
    def makeNumerical(self):
        """
        Converts/Maps all the string values in the dictionary list to float values.
        """
        for i in xrange(len(self.input_data)):
            for key in self.input_data[i].keys():
                if key in non_numeric_cols: #When the column contains non-numeric values
                    self.input_data[i][key] = self.non_numeric_maps[key][self.input_data[i][key]] #Storing non-numeric entries as integer values by using the mappings previously generated
                else: #When the column contains numeric values
                    tmp = self.input_data[i][key].split() #Split the string value into constituent parts sperated by space
                    if len(tmp) > 0:  #If the entry is not empty
                        try:
                            self.input_data[i][key] = float(tmp[0]) #When the first part is the data value/number
                        except: #To handle cases when the first part of the string is not a number
                            try:
                                self.input_data[i][key] = float(tmp[0][:-1])
                            except:
                                self.input_data[i][key] = 0. #If everything fails, assign a value of 0
                    else:
                        self.input_data[i][key] = 0. #Assign 0 value to empty entries

    
    def frange(self, y, x=0, jump=1):
        """
        Custom method to generate a list of floats instead of integers generated by the 
        range() function.
        """
        return [1. * k for k in xrange(x, y, jump)]