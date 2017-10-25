
# Requires Changes
#### 4 specifications require changes

Dear student,
First of all, thank you for your submission! It’s a very solid one! 
You did a very good job in this submission, but we still have a couple of issues to address before going ahead.
 Please refer to the comments below for details on the required changes and suggestions to further improve certain aspects of the report and code.
 I really hope you find them useful!

Keep up the great work! We are looking forward to your next submission! 😃

#### REGARDING YOUR COMMENT

The final classifier showed a very different performance when using the provided tester.py. I think the culprit is the different cross validation techniques (StratifiedKFold vs. StratifiedShuffleSplit).

Yes! That's very possible! Despite both methods being stratified, StratifiedKFold and StratifiedShuffleSplit work differently. Let's take a look at the definition of each one obtained in the original documentation:

StratifiedKFold: This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.

StratifiedShuffleSplit: This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class.

It is important to note that the first splits the data set into k folds, and loops over these folds.
The second, will repeat a holdout (stratified train_test_split) evaluation multiple times, but it doesn't a priori split the data set into k folds. It simply shuffles the data set, selects a percentage of the data set for training and the rest for training (while keeping it stratified).

*Thanks for the explanation, I'll study the possibility of changing my CV method as shuffling seems a good way to counteract the small size of the dataset.*

## Quality of Code

#### Code reflects the description in the answers to questions in the writeup. i.e. code performs the functions documented in the writeup and the writeup clearly specifies the final analysis strategy.

This item will meet specifications as soon as the items marked below also do.

*Ok, I'll fix them.*

#### poi_id.py can be run to export the dataset, list of features and algorithm, so that the final algorithm can be checked easily using tester.py.

## Understanding the Dataset and Question

#### Student response addresses the most important characteristics of the dataset and uses these characteristics to inform their analysis. Important characteristics include:

    total number of data points
    allocation across classes (POI/non-POI)
    number of features used
    are there features with many missing values? etc.

SUGGESTION

Even though this is listed later in the report, please make sure we also mention the number of POIs right here in the beginning of the report.

*Ok, done.*

#### Student response identifies outlier(s) in the financial data, and explains how they are removed or otherwise handled.
SUGGESTION

In addition to TOTAL, we have another two outliers which are pretty clear and easy to find:

    One is not a real person (take a look at the names and you will find it!)
    The second has all of its values missing!

*Ok, done.*

## Optimize Feature Selection/Engineering

#### At least one new feature is implemented. Justification for that feature is provided in the written response. The effect of that feature on final algorithm performance is tested or its strength is compared to other features in feature selection. The student is not required to include their new feature in their final feature set.
SUGGESTION

To further improve this section, it would be nice to add a small paragraph discussing how good or bad each new feature is given their f-scores.

SUGGESTION - WHY EVALUATING F-SCORES IS NOT ENOUGH

Unfortunately, evaluating the scores is not the most appropriate way to separate features. F-scores measure the separability of classes along a single feature, however, it doesn't account for combinations of features when measuring separabilities (something called feature interaction). Since classifiers work on top of all the features we feed them, it becomes the best way to evaluate a subset of features since it accounts for interactions between them.
I would also suggest taking a look at this great paper, which exemplifies how features that seem uncorrelated with the class can be found to be very important when combined with other features. Therefore, the most appropriate way to determine the real impact of new features it to assess their impact on recognition rates, it is, accuracy, precision and recall.

*Changed the feature selection method for a more robust one, using not only univariate, but recursive, hand picked and live model testing too.*

#### Univariate or recursive feature selection is deployed, or features are selected by hand (different combinations of features are attempted, and the performance is documented for each one). Features that are selected are reported and the number of features selected is justified. For an algorithm that supports getting the feature importances (e.g. decision tree) or feature scores (e.g. SelectKBest), those are documented as well.
REQUIRED

We have a solid start here, good job!
The report mentions that SelectKBest has been used to rank features. Features are then listed along with their f-scores and p-values.
Yet, to meet this criterion, we need to further clarify the feature selection process adopted. Why keeping features with a p-value below 0.001 chosen?
Here, we need to provide some evidence on why 0.001 is indeed a great choice compared to others.
Also, why weren't the f-scores used to guide the selection process?

*Changed the feature selection method for a more robust one, using not only univariate, but recursive, hand picked and live model testing too.*

#### If algorithm calls for scaled features, feature scaling is deployed.

## Pick and Tune an Algorithm

#### At least two different algorithms are attempted and their performance is compared, with the best performing one used in the final analysis.

#### Response addresses what it means to perform parameter tuning and why it is important.
REQUIRED

This rubric item requires the report to present a brief discussion about what it means to tune classifiers, its goal, and why it is important.
Even though this section already mentions what tuning is and how it works, we need to further detail each of the following aspects here:

    What is the main goal of tuning a classifier? It is, what exactly do we expect when testing different parameters?
    Is there any classifier that could fail if we only run it with the default parameters?
    What do we want to achieve by testing different values in several parameters?

*Added more depth to the explanation of parameter tuning, I hope it's enough.*

#### At least one important parameter tuned with at least 3 settings investigated systematically, or any of the following are true:

    GridSearchCV used for parameter tuning
    Several parameters tuned
    Parameter tuning incorporated into algorithm selection (i.e. parameters tuned for more than one algorithm, and best algorithm-tune combination selected for final analysis).

## Validate and Evaluate

#### At least two appropriate metrics are used to evaluate algorithm performance (e.g. precision and recall), and the student articulates what those metrics measure in context of the project task.

#### Response addresses what validation is and why it is important.

#### Performance of the final algorithm selected is assessed by splitting the data into training and testing sets or through the use of cross validation, noting the specific type of validation performed.
REQUIRED

In the report, it is mentioned that StratifiedKFold has been used to validate the tested classifiers.
This is a great approach, but I couldn't find this actual implementation on the code.
To meet this criterion, we need to make sure that the report and code match!

*Changed the Cross Validation method, explained why*

#### When tester.py is used to evaluate performance, precision and recall are both at least 0.3.
OUTPUT

    VotingClassifier(estimators=[('svc', SVC(C=11.6, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1.17, kernel='sigmoid',
    max_iter=-1, probability=False, random_state=42, shrinking=True,
    tol=0.001, verbose=False)), ('ada', AdaBoostClassifier(algorithm='SAMME',
    ...estimators=9, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False))],
            flatten_transform=None, n_jobs=1, voting='hard', weights=None)
        Accuracy: 0.81233    Precision: 0.36679    Recall: 0.56100    F1: 0.44357    F2: 0.50728
        Total predictions: 15000    True positives: 1122    False positives: 1937    False negatives:  878    True negatives: 11063

