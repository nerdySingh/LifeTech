"""
Contains default names of pickle file names which will be used to name dumped pickle files when the
dumpPickleData() method is called in the DataReader class.
"""

"""
File containing the cleaned and scraped version of the input CSV file data.
The data pertaining to each column is stored as a float.
"""
default_input_data_filename = 'input_data.dat' 

"""
File containing the labels (0 or 1) pertaining to each entry.
Here 1 stands for the "Fully Paid" label and 0 for everything else.
"""
default_label_filename = 'labels.dat'

"""
File storing the maps and relations needed to convert non-numeric entries into 
their equivalent numerical representations needed for training.
"""
default_non_numeric_maps_filename = 'non_numeric_maps.dat'