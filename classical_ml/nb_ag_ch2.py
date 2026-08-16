#%%
print("Welcome to chapter 2 of Ageron's Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow!")
#%%
import sys
assert sys.version_info >= (3, 7), "Python version must be >= 3.7"
# %%
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
#%%
import pandas as pd
#%%
import kagglehub
def load_housing_data():
    # Download the latest version of a housing dataset from Kaggle
    path = kagglehub.dataset_download("camnugent/california-housing-prices")
    print("Path to dataset files:", sys.path)
    # Load the CSV file into a Pandas DataFrame
    return pd.read_csv(path + "/housing.csv")

housing = load_housing_data()

# below is code for data on local machine
# data_path = '../data/housing.csv'
# housing = pd.read_csv(data_path)
housing.describe()  # describe() method shows a quick statistic summary of the data
housing.head()
#%%
housing.info() # The info() method is useful to get a quick description of the data, in particular the total number of rows, each attribute’s type, and the number of non-null values
housing["ocean_proximity"].value_counts() # find out what categories exist and how many districts belong to each category by using the value_counts() method
# %%
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize=(24,16)) # call the hist() method on the whole dataset and it will plot a histogram for each numerical attribute
plt.show()
#%%
# Creating a test set is theoretically simple; pick some instances randomly,
# typically 20% of the dataset (or less if your dataset is very large), and set them aside:
 
import numpy as np
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) #randomizes the indices of data
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] 

train_set, test_set = shuffle_and_split_data(housing, 0.2) # this function will create different test and train set every single time the program is run
len(train_set)
#%%
len(test_set)

# %%
# Well, this works, but it is not perfect: if you run the program again, it will
# generate a different test set! Over time, you (or your machine learning
# algorithms) will get to see the whole dataset, which is what you want to avoid

# To have a stable train/test split even after updating the dataset, a
#common solution is to use each instance’s identifier to decide whether or not 
# it should go in the test set (assuming instances have unique and immutable identifiers)
# Here is a possible implementation
from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64 (identifier)) < test_ratio *2*32
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data .loc[~in_test_set], data.loc[in_test_set]

# Unfortunately, the housing dataset does not have an identifier column
# so we have to create one by converting any column of the data to index such that each district has a unique identifier and 
# that identifier is stable across dataset updates. for example a district’s latitude and longitude are guaranteed to be stable for a few million years, so you could
# combine them into an ID
housing_with_id = housing.reset_index() # adds an index columne
housing_with_id ["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
# %%
len(train_set)
# %%
len(test_set)
# %%
 # scikit provides a few fnxs to split datasets into subsets in mulltiple ways. The simplest one is train_test_split(), 
 # which does exactly what its name suggests: it splits the dataset into a training set and a test set, and it even shuffles the data for you. 
 # Here is how to use it:
from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing, test_size=0.2, random_state=42) 
# random_state is the seed of the random number generator, so that you get the same split every time you run the code
# you can pass it multiple datasets with an identical number of rows, and it will split them on the same indices
len(train_set)

# %%
# example for stratified sampling based on the median income attribute
# The following code uses the pd.cut() function to create an income category attribute with five categories (labeled from 1 to 5);
# category 1 ranges from 0 to 1.5 (i.e., less than $15,000), category 2 from 1.5 to 3, and so on:

housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot =0,grid = True)
plt.xlabel("Income Category")
plt.ylabel("Number of Districts")
plt.title("Districts by Income Category")
# %%
# Now that you have created the income category attribute, you can use Scikit-Learn’s StratifiedShuffleSplit class to do a stratified sampling based on this attribute.\
# to be precise, the split() method yields the indices of the training set and test set, not the data itself.
# splitting dataset into 10 stratified splits
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state = 42)
strat_splits =[]
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_test_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_test_n,strat_test_set_n])
# now you can use the first split
strat_test_set, strat_train_split = strat_splits[0]


# %%
# instead of all this we can do the following
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, random_state =42, stratify = housing["income_cat"])
# let's see if all of this works 
strat_test_set["income_cat"].value_counts()/ len(strat_test_set)
# %%
# you might never use the "income_cat" attribute again, so you should drop it to revert data to its original form.
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
# %%
