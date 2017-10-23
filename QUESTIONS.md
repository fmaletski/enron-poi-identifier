### **Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]**

The famous ENRON scandal was the largest bankruptcy reorganization in the United States at the time it was publicized, October 2001. Due to the Federal investigation, a significant amount of confidential information was released to the public, including tens of thousands of emails and detailed financial data. The objective of this project is to use this large dataset to create a machine learning model that correctly identifiers the Persons of Interest (POI) based on the data made public.

The dataset contained an outlier, called 'TOTAL'. It was removed before proceeding with the analysis. 

With the outlier removed, there were 145 persons in the dataset, with 21 features each, including the POI flag.

### **What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]**

#### Feature creation and removal
I started by analyzing the email features and decided to create 2 new features based on ratios:
* 'from_poi_ratio' - ratio of messages from POI of all the received messages
* 'to_poi_ratio' - ratio of messages to POI of all the sent messages

They were created because there were people who sent a lot of emails and those that didn't, so reducing it to ratios makes sense for comparison. Both were selected to proceed with the statistical features selection, but only 'to_poi_ratio' was used in the creation of the model, 'from_poi_ratio' was deemed irrelevant.

Of the financial features 4 were removed due to excessive missing values and not enough POI (loan_advances, director_fees, restricted_stock_deferred, and deferral_payments):

                         Title: Count    Missing  POI Count  % Missing
                 loan_advances: 3        142      1          97.93   
                 director_fees: 16       129      0          88.97   
     restricted_stock_deferred: 17       128      0          88.28   
             deferral_payments: 38       107      5          73.79   
               deferred_income: 48       97       11         66.90   
           long_term_incentive: 65       80       12         55.17   
                         bonus: 81       64       16         44.14   
                         other: 92       53       18         36.55   
                        salary: 94       51       17         35.17   
                      expenses: 94       51       18         35.17   
       exercised_stock_options: 101      44       12         30.34   
              restricted_stock: 109      36       17         24.83   
                total_payments: 124      21       18         14.48   
             total_stock_value: 125      20       18         13.79   

#### Feature selection

To select the features to be used in the creation of the model, I used SelectKBest with the standard ANOVA function. Here's the resulting table:

                       p_value :           Feature            : F-score

         1.398443796240722e-06 :   exercised_stock_options    : 25.3801052997602
         1.844426415180382e-06 :      total_stock_value       : 24.752523020258508
         8.548240899761827e-06 :            bonus             : 21.3278904139791
        2.6464485993698053e-05 :            salary            : 18.861795316466416
          6.69501897034152e-05 :         to_poi_ratio         : 16.873870264572993
         0.0008017876244783438 :       deferred_income        : 11.732698076065354
           0.00170779873759263 :     long_term_incentive      : 10.222904205832778
          0.002490134850136291 :       restricted_stock       : 9.480743203478934
          0.003239087410989142 :        total_payments        : 8.96781934767762
          0.012667743910203918 :           expenses           : 6.3746144901977475
           0.04074414136894844 :            other             : 4.263576638144469
            0.0716363587097375 :        from_poi_ratio        : 3.293828632029562

I decided to select features with a p_value of less than 0.001, therefore, the top 6 features (k=6) were used.

#### Feature scaling

I was planning on using SVM so I scaled the features using MinMaxScaler. It scales all the features to values between 0 and 1.

This increases the performance of SVM based algorithms, both in running speed and end results.

### **What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]**

To choose an algorithm, I followed the Sklearn directives found here:  http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

Starting with the simplest all the way to ensemble classifiers:


    Linear SVC
    KN Classifier
    SVC (other kernels)
    Ensemble Classifiers
        - Random Forest
        - Adaboost

I did not test the performance of them beforehand, I proceeded directly to cross validation and parameter tuning to choose the final one.

### **What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]**

Parameter tuning is where the art of machine learning resides. Picking an algorithm "off the shelves" rarely will result in optimal performance.

Each algorithm has different parameters than in combination can dramatically change how it performs. Here's the list of the tuned parameters:

* Linear SVC
    + C
    + class_weight

* KN Classifier
    + n_neighbors
    + weights
    + algorithm
    + leaf_size
    + p

* SVC (other kernels)
    + kernel
    + C
    + gamma
    + class_weight

* Decision Tree
    + criterion
    + max_features
    + min_samples_leaf
    + class_weight

* Random Forest
    + n_estimators
    + criterion
    + max_features
    + min_samples_leaf
    + class_weight

* Adaboost
    + base_estimator
    + n_estimators
    + learning_rate
    + algorithm

The first step in fine tuning the models was choosing the right evaluation metric and cross validation strategy for this dataset.

#### Evaluation metrics

The dataset is very unbalanced towards non-POI:
POI: 18 | Total: 145
Accuracy if predicted all non-POI: 0.875862

Due to the imbalanced nature of the dataset (way more non-POI than POI), using just accuracy, or even F1, results in poor detection performance. The objective here is fraud detection! A model that is accurate but doesn't detect a lot of POI is not a good one.

There is a metric specifically created to deal with highly imbalanced classes, called Matthews correlation coefficient:

The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary (two-class) classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.[source: Wikipedia | http://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html | https://en.wikipedia.org/wiki/Matthews_correlation_coefficient]

The MCC is the chosen metric in this project for parameter tuning and evaluation.

### **What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]**

Evaluating the performance of an algorithm without proper validation can result in overfitted models. This happens when the model performs great in during training, but does not have enough variability to predict new data. To avoid this, I implemented cross validation.

The method chosen to it was the StratifiedKFold (http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) because of the class imbalance. This preserves the percentage of samples for each class. And due to the small size of the dataset (only 145 datapoints), I chose to use just 3 folds. In other words, 66.6% of the dataset was used to train the model and 33.3% to test it and this was repeated 3 times and the results averaged.

### **Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]**

The metrics chosen to evaluate each model were:

* Matthews correlation coefficient: 

A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.

* F1-score

The weighted average of the precision and recall.

* Precision

The ratio between true positives and all the positives predicted by the model. This can be interpreted as if the model accuses a person of being a POI, how sure is it.

* Recall

The recall is the ability of the classifier to find all the positive samples. Of all the POI what is the ratio found by the model.

* Accuracy

How accurate is the predictions, both positive and negative.

#### Performance

After exhaustively testing and parameter tuning, here are the models ranked, by the Matthews Correlation Coefficient:

               Classifier   MCC        F1    Precision   Recall   Accuracy 

             RandomForest  0.5411    0.5394    0.7759    0.5000    0.9103  
                 AdaBoost  0.5379    0.5942    0.5425    0.6667    0.8897  
                      SVC  0.5053    0.5287    0.3750    0.9448    0.7793  
                LinearSVC  0.4080    0.4515    0.4937    0.6690    0.7310  
             DecisionTree  0.3579    0.4473    0.3808    0.5552    0.8207  
               KNeighbors  0.0607    0.1411    0.1948    0.1115    0.8345  

The top three models have each showed a useful trait:

    Random Forest: Best precision
    Adaboost: Best F1 (balance between precision and recall)
    SVM: Best recall

The first two also got a very good accuracy score that trumps the score of just classifying every person as non-POI, 0.8760.

Using a voting classifier enables the models to achieve a performance that none of them could on their own. Each classifier have one vote, and the predicted class is determined my the majority.

http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

The scores were:

    MCC:       0.7220
    F1:        0.7498
    Precision: 0.7008
    Recall:    0.8333
    Accuracy:  0.9310

The end result is a classifier that have a lower precision than the Random Forest, and a lower recall than the SVC, but there is value in balance.

It achieved the highest Accuracy, F1 and MCC scores by far! And because of that, it is the chosen model and the end result of this project.