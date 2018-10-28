# Machine Learning
Projects of CS-433 - Machine Learning, EPFL (Fall 2018).

_Class Project 1_

_Group : Gaia Carparelli, Hugues Vinzant & Axel Bisi_


#### RUNNING INSTRUCTIONS:

To run this machine learning system, please proceed along the steps:

	Step 1: Download the code.zip file from the online submission system (https://epfml17.hotcrp.com). The .zip file 	        contains an executably run.py, an implementations.py file and proj1_helpers.py file.
	
	Step 2: Make sure that .csv data files are in the same folder as the file run.py. 
		If the names of the .csv data files differ from the ones present in the part "----  LOAD THE DATA  ----" of 	    	    run.py, change the names accordingly.

	Step 3: Now, to run the code from the shell type:  python run.py 

	Step 4: Wait for the execution until "Execution completed." appears on your terminal.
	
	Step 5: The submission .csv file for evaluation may be found in the same folder.


#### REPRODUCING RESULTS:

The executable run.py contains the final model (model B) which produces the best classification performance. It is our submitted model. However, to reproduce results of the other models(models A, C & D), one must decomment respectively parts of the file run.py, while commenting the other parts.

- To obtain model A, decomment Part A or run.py and comment all others parts, that is Part B, Part C and Part D. 
Type on the shell : python run.py

- To obtain model B, decomment Part B or run.py and comment all others parts, that is Part A, Part C and Part D. 
Type on the shell : python run.py

- To obtain model C, decomment Part C or run.py and comment all others parts, that is Part A, Part B and Part D. 
Type on the shell : python run.py

- To obtain model D, decomment Part D or run.py and comment all others parts, that is Part A, Part B and Part C. 
Type on the shell : python run.py

#### CODE ARCHITECTURE:

The code is composed of the following files:

 - run.py
 
 CONTENT: Contains the full executable code performing the loading, preprocessing of the data, training and cross-validation of the model, and the prediction and classification. This file imports the functions of all the files implementations.py and proj1_helpers.py. 

 - implementations.py
 
 CONTENT: Contains all the functions of the project which we implemented. For more details about each function, the reader is redirected to the file and function in question. The file is divided into three parts:
 
 	- Part IMPLEMENTATIONS: Contains the 6 required methods, that is, least_squares_GD, least_squares_SGD, least_squares, 				  ridge_regression, logistic_regression, reg_logistic_regression. Not all of them are used.
	
	- Part UTILITARIES: Contains all functions required for the functions in IMPLEMENTATIONS, such as gradient calculations, loss calculations, learning algorithms, score predictions, etc.  
	
	- Part DATA PROCESSING: Contains all functions used for data pre-treatments, such as standardization, variance 				      calculations, data splitting, removal of data, estimation of data, etc.
	
	
- proj1_helpers.py

CONTENT: Contains given functions for loading the data, prediction labels and creating a submission in .csv format.



