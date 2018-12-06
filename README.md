### Notes:
Provided scripts construct the baseline of this project. It is able to produce the prediction results, but you will need to modify them (and add new ones if necessary) to optimize the solution.
1. A symbolic link “data” should be created pointing to the folder that stores dependencies (.txt and .csv files). 
3. Intermediate files are saved into this "data" folder as well. Change the codes accordingly if you want them to be saved somewhere else.
4. Because this dataset has large size, xgboost may take many hours and a lot of memory to run. You may want to test the pipeline with a small portion of dataset first before using all of them. Different parameters for xgboost can also be experimented. 

### Description:
Sample count in SNP file: 20087
Sample count in phenotype file: 20723
Intersection: 19534

17699 samples have oil production in phenotype record, and 124 amoung them are not in SNP file. Therefore, number of samples that are put into xgboost is 17575.

### Files:
plot.py: to plot distributions of traits;
preprocessing.py: to extract oil production from phenotype records, and SNP sequences from SNP records. They are saved to dictionaries as intermediate results  (because this process can be very time-consuming);
xgb.py: to use extracted SNP information as features, and oil information as label to predict oil production for a given SNP sequence. Cross-validation mean square error scores are reported.
