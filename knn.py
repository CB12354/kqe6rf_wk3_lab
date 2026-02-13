#%% imports
import numpy as np
import pandas as pd
import seaborn as sns
from datapipeline import *

from sklearn.neighbors import KNeighborsClassifier

# %% [markdown] The Problem
# 1. Use the question/target variable you submitted and build a model to 
# answer the question you created for this dataset (make sure it is a 
# classification problem, convert if necessary). 
#
# The problem I came up with was trying to predict if a school was likely
# to give a merit scholarship based on other statistics. There's a column
# called "is_generous" that's the target.
# The model is a nearest neighbors model that tries to predict is_generous
# based on other metrics about the school.
#
# My models are coded below as part of questions 2 onward

#%% 
# 2. Build a kNN model to predict your target variable using 
# 3 nearest neighbors. Make sure it is a classification problem, meaning
# if needed changed the target variable.

#%% load transformed dataset
college_f = format_college_dataset()
college_t = transform_college_dataset(college_f)
# %% tts and split features from target
train, test = triple_t_split(college_t, .7, "is_generous", tune = False)

X = train[[col for col in train.columns if col != 'is_generous']]
Y = train[['is_generous']]


# %% Create model
model = KNeighborsClassifier(n_neighbors = 3)
fitted_model = model.fit(X, Y)

# %%
# 3. Create a dataframe that includes the test target values, test 
# predicted values, and test probabilities of the positive class.

# Column 1: test targets (test Y)
# Column 2: test predictions (predict())
# Column 3: test probabilities (predict_proba())
X_test = test[[col for col in train.columns if col != 'is_generous']]
Y_test = test['is_generous']
predictions = fitted_model.predict(X_test)
probs = fitted_model.predict_proba(X_test)[:,1]

df_p = pd.DataFrame(data={'actual':Y_test, 'predicted':predictions, 'probs': probs})
print(model.score(X_test, Y_test))
# %% CM
pd.crosstab(df_p['actual'], df_p['predicted'])

# %%
#4. No code question: If you adjusted the k hyperparameter what do you think
# would happen to the threshold function? Would the confusion look the same 
# at the same threshold levels or not? Why or why not?
#
# The threshold function wouldn't alter its simple majority classification rule,
# but it would change its results because it is looking at more 
# neighboring points to make the choice for which group a point goes to. 
# The confusion matrix wouldn't be the same since it has to adjust 
# to the new neighbors. (ex. 3 neighbors could classify to one group, while 5
# could classify to the other)

# %%
# 5. Evaluate the results using the confusion matrix. Then "walk" through 
# your question, summarize what concerns or positive elements do you have 
# about the model as it relates to your question?
# 
# The model gets about 65% of the "generous" schools correct (54/83) and
# about 90.2% of the "non-generous" schools correct (591/655). My question
# was about trying to see if a school was likely to give a merit scholarship
# given other statistics about it. This model is easy to set up and can log 
# and predict from this few points very quickly However, its accuracy at 
# k=3... leaves something to be desired. The prevalence of generous schools
# is very small, so we'd likely need more schools or a wider definition
# of "generous" to improve accuracy. Either that, or the other factors may
# not do a good job of predicting this.

# %%
# 6. Create two functions: One that cleans the data & splits into 
# training|test and one that allows you to train and test the model with 
# different k and threshold values, then use them to optimize your model 
# (test your model with several k and threshold combinations). Try not to 
# use variable names in the functions, but if you need to that's fine. (If 
# you can't get the k function and threshold function to work in one 
# function just run them separately.) 

def score_with_threshold(probs, actual, thresh: float = 0.5) -> float:
    classes_cast = [1 if prob >= thresh else 0 for prob in probs]
    correct = list(np.zeros(len(classes_cast)))
    actual = list(actual)
    for i in range(len(classes_cast)):
        correct[i] = (actual[i] == classes_cast[i])
    #print("Correct after:")
    #print(correct)
    return (sum(correct) / len(correct))
    

def clean_split_college(target: str):
    return triple_t_split(transform_college_dataset(format_college_dataset()),
                          .7, target, tune = False)

def train_test_model_college(train: pd.DataFrame, test: pd.DataFrame, k: int, target="", thresh: float = 0.5):
    X = train[[col for col in train.columns if col != target]]
    Y = train[[target]]
    model = KNeighborsClassifier(n_neighbors = k)
    fitted_model = model.fit(X, Y)
    
    X_test = test[[col for col in train.columns if col != target]]
    Y_test = test[target]
    predictions = fitted_model.predict(X_test)
    probs = fitted_model.predict_proba(X_test)[:,1]

    df_p = pd.DataFrame(data={'actual':Y_test, 'predicted':predictions, 'probs': probs})
    return score_with_threshold(df_p['probs'], Y_test, thresh=thresh), df_p

# %% Test different thresholds and k values
np.random.seed(125)
train, test = clean_split_college("is_generous")
kgrid = [2*k + 1 for k in range(0,50)]
acc = []
threshes = [0.1*x for x in range(1,10)]
for thresh in threshes:
    accrow = []
    for k in kgrid:
        score = train_test_model_college(train, test, k, target='is_generous', thresh = thresh)[0]
        print(score)
        accrow.append(score)
    acc.append(accrow)

# %% Find the best threshold index (0.1 to 0.9 from indexes 0 to 8)
for row in range(len(acc)):
    sns.lineplot(x=kgrid, y=acc[row])
    print(max(acc[row]))
# Best threshold = 30 or 40% - have equal maxes. 
# Choosing 30%, which is at index 2

# Find the best k in that row
print(kgrid[np.argmax(acc[2])])
# best k = 7


# %%
# 7. How well does the model perform? Did the interaction of the adjusted thresholds and k 
# values help the model? Why or why not? 
# Given optimized parameters, score is 90.5% - tuning helped improve the overall score 
# a little bit (improved the score by ~3%), but breaking it down by type of result...
df_results = train_test_model_college(train, test, 7, target='is_generous', thresh = 0.3)[1]
df_results = df_results.reset_index(drop=True)
df_results['predicted'] = df_results['probs'].apply(lambda x: x >= 0.3)
df_not_gen = df_results[df_results['actual'] == False]
df_gen = df_results[df_results['actual'] == True]
ratio_not_gen = sum(df_not_gen['actual']==df_not_gen['predicted'])/len(df_not_gen)
ratio_gen = sum(df_gen['actual']==df_gen['predicted'])/len(df_gen)
# ...we see that the predictive power for the positive class is actually 
# worse! (62.7%, down from 65%) The predictive power for the negative class
# went up, though (95.8% up from 90.2). Looks like this model wasn't helped much
# by the tuning.

# %%
# 8. Choose another variable as the target in the dataset and create another kNN model using the two functions you created in
# step 7. 

# Trying to predict if flagship or not
np.random.seed(125)
train_ctrl, test_ctrl = clean_split_college("flagship")
score_ctrl = train_test_model_college(train_ctrl, test_ctrl, 3, target='flagship')[0]
print(score_ctrl)
# %%
# Looks like this model can predict quite accurately already 
# with k=3 and threshold = 50%, score of 98%