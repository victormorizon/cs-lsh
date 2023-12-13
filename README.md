# cs-lsh

## What is this about?

This code can be used to detect duplicates whithin a dataset of 1624 TV product descriptions coming from 4 webshops: Amazon, The Nerds, Best Buy and Newegg.
This uses MinHash, LSH and a Random Forest classifier in order to achieve that.

## How is this structured?
* Both data files (product details and reference brand) are located in the Data folder.
* The Images folder contains 3 plots that can be used to analyse the performance of the model
* classes_functions.py contains all the functions needed in this program (from MinHash to LSH and classifier)
* main.ipynb is the file to run. It imports all functions from classes_functions.py, preps the data and runs the full model

## How to run it?
1. Open the main.ipynb and run the first 3 cells (import, load and prep)
2. Specify how many runs you want to have (and whether you want to run the model for multiple number of LSH bands). Be careful, in its current form, the code will run the model a total of 760 times, taking a few hours to complete. 
3. If you don't want to run it for multiple bands, then comment out the first for-loop (and the last append statements) and print out the median for f1, f1_star and fraction of comparisons.
4. The final cell can be used for plotting across different fractions of comparisons. Note that the plotting_data has been pre-saved (as it takes a few hours to run). If you want to re-run it, you'll have to make your new plotting_data dataframe in the same format as the current one.