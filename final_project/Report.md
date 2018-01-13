# Machine Learning Project: Identify Fraud from Enron Email
## Udacity Data Analyst Nanodegree
By Jay Stoltzenberg

### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of the project is to identify employees from Enron who may have committed fraud based on the public Enron financial and email dataset, within this project we call them persons of interest (poi). Persons of interest are defined as individuals who have been indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

In identifying these poi, machine learning algorithms are a very useful tool because they can process datasets way faster than humans and can spot relevant patterns in the data that we would have a hard time realizing manually. 

Let us begin by looking at some background on the Enron financial and email dataset:

#### Employees

There are 146 Enron employees in the datasetof which 18 are POIs. This leaves 128 non-POIs.

#### Features

The dataset holds a total of 21 features. 14 of them are financial features and listed below. All units are US dollars.

- salary
- deferral_payments
- total_payments
- loan_advances
- bonus
- restricted_stock_deferred
- deferred_income
- total_stock_value
- expenses
- exercised_stock_options
- other
- long_term_incentive
- restricted_stock
- director_fees

There are also six email related features. All units are number of emails messages, except for ‘email_address’, which is a text string.

- to_messages
- email_address
- from_poi_to_this_person
- from_messages
- from_this_person_to_poi
- shared_receipt_with_poi

Finally, there is one boolean feature, indicating whether or not the employee is a person of interest.
 
- poi

#### Missing Values

With the exception of POI all features have have missing values ("NaN"). A summary of the missing values can be seen below:

- poi 0
- salary 50
- deferral_payments 106
- total_payments 21
- loan_advances 141
- bonus 63
- restricted_stock_deferred 127
- deferred_income 96
- total_stock_value 19
- expenses 50
- exercised_stock_options 43
- long_term_incentive 79
- restricted_stock 35
- director_fees 128
- to_messages 58
- from_poi_to_this_person 58
- from_messages 58
- from_this_person_to_poi 58
- shared_receipt_with_poi 58
- bonus_salary_ratio 63
- from_this_person_to_poi_percentage 58
- from_poi_to_this_person_percentage 58

Of these features I have decided not read in "other" and "email_address" as I can´t imagine them bringin much value

#### Outliers

To identify outliers I plotted the salary over the bonuses and it is easy to identify an outlier already seen throughout the course.It is the "TOTAL" entry and it is much higher in both salary and bonus than the rest of the entries. this also is seen in other financial features of the entry. This a non-person entry that results from erroneous data entry from the total of an input list. In addition I found another non-person entry that should be removed it is called 'THE TRAVEL AGENCY IN THE PARK'. Both entries are removed from the dataset. 

### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]



#### Features Engineered

I have added three features to the dataset:

- bonus_salary ratio
- from_this_person_to_poi_percentage
- from_poi_to_this_person_percentage

"Bonus_salary_ratio" is intended to highlight people who have low salary and high bonuses as they may indicate mischief.
Furthermore the features "from_this_person_to_poi_percentage" and "from_poi_to_this_person_percentage"

#### Feature Selection 

Feature selection methods can be used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model.
Fewer attributes is also desirable as because it reduces the complexity of the model, and a simpler model is simpler to understand and explain.
(https://machinelearningmastery.com/an-introduction-to-feature-selection/)

Before going into modeling I will do a preliminary feature selection using the k best method. In default mode this finds the k best features by ranking the TANOVA F-value between label/feature for classification tasks. The results on our features are as follows

|Rank|**K-best features:**|
|---|---|
|1|(True, 'exercised_stock_options', 24.815079733218194)|
|2|(True, 'total_stock_value', 24.182898678566879)|
|3|(True, 'bonus', 20.792252047181535)|
|4|(True, 'salary', 18.289684043404513)|
|5|(True, 'from_this_person_to_poi_percentage', 16.409712548035792)|
|6|(True, 'deferred_income', 11.458476579280369)|
|7|(True, 'bonus_salary_ratio', 10.783584708160824)|
|8|(False, 'long_term_incentive', 9.9221860131898225)|
|-|...

With this result I have decided to include all features above an importance rating of 10. This leaves us with 7 featres. And it´s great to see that 2 of our new features made the cut :)

We will leave it up to the frid search later own to see if selection of even fewer variables leads to better results.

#### Scaling

We will use MinMax scaling where needed. Whether we need to use it depends on the algorithm. Tree based algorithms use threshold values for each feature and are thus not prone to large scales in between features.
However k nearest neighbors and SVM need to have scaling added as the difference between points when clustering affect the outcome of the clusters. a features with much higher values will thus domicate the clustering. therefore a scaling of continuous features from 0 to 1 is important here.

With this in mind we will use scaling for k nearest neighbors and SVM. For Decision trees and and random forests scaling will not be needed. 


### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I have decided to try four algorithms and employ a grid search for each of them to find the parameters that yield the best results.

The algorithm used are:

- decision tree classifier
- random forests
- SVM
- k-nearest neighbors

By created pipelines for each algorithm giving several options for the most important paramter. The results for precision and Recall can be found in the table below:

|Classifier|Precision|Recall
|---|---|---|
|Random Forest|**0.34918**|0.60600|
|Decision Tree|0.32778|**0.63000**|
|SVC|0.61003|0.10950|
|K Nearest Neighbor|0.20717 |0.19350|

It is not surprising that the SVM classifier performs badly as the poi label is quite unbalanced and easily overfits to the points in the training data.

I also expected k nearest neighbor to perform somewhat better as I would think pois would employ similar behaviour and would lie close togehter.

The decision tree performs quite well and thus it only makes sense to also try a random forest model as decision trees tend to overfit the data whereas random forests employ multiple decision trees and vote among trees to achieve a model less prone to overfitting and more balanced. However it is not easily possible to identify what features were given the highest importance in a random forest as each decision in the forest may have different structures.

Therefore I will show the chosen features and their importances for the case of using the decision tree:

The  5  features selected and their importances:

|Rank|Feature|importance from decision tree|importance from k best| 
|---|---|---|---|
|1|from_this_person_to_poi_percentage |1.0| 6.8203206006|
|2|deferred_income|0.0|13.7058385282|
|3|bonus|0.0|9.14372549617|
|4|total_stock_value|0.0|8.96461980545|
|5|exercised_stock_options|0.0|9.2156350679|

Interestingly enough the decision tree puts all the importance on the from:this_person_to_poi feature.

### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

Using the available parameters of an algorithm can be a valuable way to optimize the performance of your algorithm. While one can just play with the parameters it is often better to autimatically try different setting in a meaningful range.
GridSearchCV is a great way to automate these stapes as it allows to run the algorithm through a predefined variety of setting for each parameter and in the end chooses the parameter combination with the best score.
Badly tuned algorithms can perform significantly worse than finely tuned versions of the same algorithm. So if you don't tune the algorithm well, performance may suffer, the data won't be "learned" well and you won't be able to successfully make predictions on new data.

For the random forest algorithm I passed these possibilities for parameters into the Grdsearch pipeline. The values chosen as most successful combinations are marked in bold:


|parameter|Options|
|---|---|
|n_estimators|[5,**10**]|
|criterion|['gini', **'entropy'**]|
|max_depth|[0,**1**,2,3, 4, 5, 8]|
|min_samples_split|[**1**, 2, 3, 4]|
|class_weight|[None, **'balanced'**]|
|random_state|[**42**]|

Note that a few of these chosen parameter values were different than their default values, which proves the importance of tuning.

### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]

In validation we ensure that our machine learning model generalizes well and has a solid performance in realtion to our expectations.
Here we commonly split our data into training and testing data, while commonly learning on somewhat more of the data and testing on the rest. Without this seperation it would be hard to judge whether our algorithm could also manage input well outside of the learned data.

Fort this project we seperate the data into a test of 30% of the data and training on the remaining 70%.

To better understand whether our result qualifies the .3 expectation on precision and recall we use tester.py's Stratified Shuffle Split cross validation. Because the Enron data set is so small, this type of cross validation is useful because it creates multiple datasets out of a single one to get more accurate results.

### Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The two important evaluation metrics for this POI identifier are precision and recall. The average precision for the random forest  tree classifier was 0.34918 and the average recall was 0.60600.

Precision is how often our POI class prediction is right when we guess that class
Recall is how often we guess the POI class when the class actually occurred

When predicting persons of interest we can argue that getting the recall to a high number is most important because we don´t want to miss any POIs. So when thinking in terms of law enforcement I want to measure myself against how good I am in identifying POIs against the entire actual POI population.
https://stackoverflow.com/questions/14117997/what-does-recall-mean-in-machine-learning

Therefore as my final algorithm I will choose the decision tree algorithm as it has produced the highest recall rate with a result of 0.63000

In the end I decided to choose the random forest algorithm. While it´s scores compared to decision tree are quite similar the algorithm is more robust as explained above. But as a downside random forests are more intense in terms of computation and take lonmger to reach a result. 

And while in theory random forests should usualy be the more solid algorithm, we can also argue that it is likely not so neccesary as the enron dataset is quite specific and we don´t expect to be adding any more data to it anymore. This could look different if he had a dataset that we know is still growing over time.

### Annex: detailed Algorithm results from tester.py

Random Forest:

Pipeline(steps=[('feature_selection', SelectKBest(k=6, score_func=<function f_classif at 0x000000000D944C18>)), ('classifier', RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='entropy', max_depth=1, max_features='auto',
            max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=1,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=42, verbose=0, warm_start=False))])
        Accuracy: 0.78236       Precision: 0.34918      Recall: 0.60600 F1: 0.44306     F2: 0.52829
        Total predictions: 14000        True positives: 1212    False positives: 2259   False negatives:  788   True negatives: 9741
        
Decision Tree:

Pipeline(steps=[('feature_selection', SelectKBest(k=5, score_func=<function f_classif at 0x000000000D944C18>)), ('classifier', DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=1,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=1, min_weight_fraction_leaf=0.0,
            presort=False, random_state=42, splitter='best'))])
        Accuracy: 0.76257       Precision: 0.32778      Recall: 0.63000 F1: 0.43121     F2: 0.53191
        Total predictions: 14000        True positives: 1260    False positives: 2584   False negatives:  740   True negatives: 9416
        
SVC:

Pipeline(steps=[('min_max_scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('feature_selection', SelectKBest(k=6, score_func=<function f_classif at 0x000000000D944C18>)), ('classifier', SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
        Accuracy: 0.86279       Precision: 0.61003      Recall: 0.10950 F1: 0.18567     F2: 0.13100
        Total predictions: 14000        True positives:  219    False positives:  140   False negatives: 1781   True negatives: 11860
KNN:

Pipeline(steps=[('min_max_scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('feature_selection', SelectKBest(k=2, score_func=<function f_classif at 0x000000000D944C18>)), ('classifier', KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform'))])
        Accuracy: 0.77900       Precision: 0.20717      Recall: 0.19350 F1: 0.20010     F2: 0.19609
        Total predictions: 14000        True positives:  387    False positives: 1481   False negatives: 1613   True negatives: 10519
