"""
File contains list of main features which are extracted from the CSV fileand utilized for training & testing.
"""

"""
Columns used along with their definitions are-
funded_amnt:        The total amount committed to that loan at that point in time.
funded_amnt_inv:	The total amount committed by investors for that loan at that point in time.
out_prncp:	        Remaining outstanding principal for total amount funded
out_prncp_inv:	    Remaining outstanding principal for portion of total amount funded by investors
total_pymnt:	    Payments received to date for total amount funded
total_pymnt_inv:	Payments received to date for portion of total amount funded by investors
total_rec_int:	    Interest received to date
total_rec_prncp:	Principal received to date
"""

mainCols = ['out_prncp_inv', 'out_prncp', 'funded_amnt', 'funded_amnt_inv', 'total_rec_int', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp']

non_numeric_cols = [] #Columns needed for training but containing non numeric values like grades ('A', 'B'... etc) (Deprecated)