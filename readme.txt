Details
Code related to manuscript titled "Applying machine learning methods to predict geology using soil sample geochemistry", authored by Timothy C.C. Lui, Daniel D. Gregory, Marek Anderson, Well-Shen Lee, and Sharon A. Cowling.

Requirements
Python 3.8
scikit-learn 0.23
imblearn 0.0
numpy
pandas
matplotlib

Files
correlatedFeatures.py is used for data cleaning of correlated features.
pipelineTopography.py tests the utility of including topographic data.
pipelineSamplingMethod.py compares the various sampling methods used.
pipelineMCS.py compares the machine learning algorithms and the multiple classifier systems.
sameClassOrderNine.py is a function used in pipelineMCS.py.
TestData.csv is randomized data with same format of real data to test the functionality of the code.
