# Machine Learning
Projects of CS-433 - Machine Learning, EPFL (Fall 2018).

_Class Project 1_

_Group : Axel Bisi, Gaia Carparelli & Hugues Vinzant_


#### RUNNING INSTRUCTIONS:

To run this machine learning system, please proceed along the steps:

	Step 1: Download the code.zip file from the online submission system (https://epfml17.hotcrp.com). 
	The .zip file contains an executably run.py, an implementations.py file and proj1_helpers.py file.
	
	Step 2: Make sure that .csv data files are in the same folder as the file run.py. 
	If the names of the .csv data files differ from the ones present in the part "----  LOAD THE DATA  ----" of run.py, change the names accordingly.

	Step 3: Now, to run the code from the shell type: python run.py 
	You will be asked to choose a model.

	Step 4: Wait for the execution until "Execution completed." appears on your terminal.
	
	Step 5: The submission .csv file for evaluation may be found in the same folder as run.py.


#### REPRODUCING RESULTS:

The executable run.py contains the final model (model B) which produces the best classification performance. It is our submitted model. However, to reproduce results of the other models (models A, C & D), the user is asked which model they want to use by typing the appropriate model (case-sensitive):

- To obtain model A, execute run.py and press 'A'.
  The generated predictions can be found in the model_A.csv

- To obtain model B, execute run.py and press 'B'. 
  The generated predictions can be found in the model_B_FINAL.csv

- To obtain model C, execute run.py and press 'C'.
  The generated predictions can be found in the model_C.csv

- To obtain model D, execute run.py and press 'D'.
  The generated predictions can be found in the model_D.csv

#### CODE ARCHITECTURE:

The code is composed of the following files:

 - run.py
 
 CONTENT: Contains the full executable code performing the loading, preprocessing of the data, training and cross-validation of the model, and the prediction and classification. This file imports the functions of all the files implementations.py and proj1_helpers.py. 

 - implementations.py
 
 CONTENT: Contains all the functions of the project which we implemented. For more details about each function, the reader is redirected to the file and function in question. The file is divided into three parts:
 
 	- Part IMPLEMENTATIONS: Contains the 6 required methods, that is, least_squares_GD, least_squares_SGD, least_squares, ridge_regression, logistic_regression, reg_logistic_regression. Not all of them are used.
	
	- Part UTILITARIES: Contains all functions required for the functions in IMPLEMENTATIONS, such as gradient calculations, loss calculations, learning algorithms, score predictions, etc.  
	
	- Part DATA PROCESSING: Contains all functions used for data pre-treatments, such as standardization, variance calculations, data splitting, removal of data, estimation of data, etc.
	
	
- proj1_helpers.py

CONTENT: Contains given functions for loading the data, prediction labels and creating a submission in .csv format.



