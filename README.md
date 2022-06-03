# Machine-Learning-Feature-Selection
This project focuses on implementation of three feature selection algorithms such as (Unsupervised Discriminative Feature Selection [UDFS], Local Learning-based Clustering Feature Selection [LLCFS], Correlation-based Feature Selection [CFS]) along with 3 classifiers (Random Forest (RF), Multilayer Perceptron (MLP), k-Nearest Neighbour (k-NN)) to obtain the best possible accuracy and f1 score for given dataset which is further cross-validated using k-fold cross validation technique.

I have used the following codes to implement feature selection algorithm
and classifiers.

To execute .m files I have used matlab. 
To execute python code I have used google colab.

1. Baseline code : baseline.py
This python file contains the code that separates the dataset into testing and training data.
Post splitting the data, we have used data pre-processing to simplify
the training and testing process by appropriately transforming the whole dataset.
Once the separation is done, the classifiers(RF,KNN,MLP) are applied to calculate the 
accuracy and f-measure of respective classifiers.

2. FS10 Algo : FS10_FSAlgo.m
This matlab file contain feature selection algorithms (UDFS, CFS, LLCDFS) and provides the
ranking as an output. 

3. UDFS Algo : UDFS.m
This file contains the code of UDFS algorithm in matlab.

4. CFS Algo : cfs.m
This file contains the code of CFS algorithm in matlab.

5. LLCFS Algo : llcfs.m
This file contains the code of LLCFS algorithm in matlab.

6. FS10 Classifiers : FS10_Classifiers.py
This file uses the ranking which we got from FS10_FSAlgo.m file. Then we apply different
classifiers(RF,KNN,MLP) to calculate the the accuracy and f-measure of each algorithm.
