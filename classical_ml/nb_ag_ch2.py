#%%
print("Welcome to chapter 2 of Ageron's Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow!")
#%%
import sys
assert sys.version_info >= (3, 7), "Python version must be >= 3.7"
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
import pandas as pd
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
# CRC32 (32-bit Cyclic Redundancy Check) is a fast, deterministic checksum algorithm used 
# to detect accidental data corruption, transmission errors, and bitrot in files, network packets, and storage.
# Used in databases and software code to rapidly compare if two data strings or records match without comparing the full data payload
def test_set_check(identifier, test_ratio): # takes the row identifier and converts it into a 64-bit integer
    return crc32(np.int64 (identifier)) < test_ratio *2**32  
# it calculates the CRC32 checksum of the integer.
# an dthen check if the hash  is less than the 20 percent thereshold of all possible hash values 
# it returns True if the row should be in the test set, and False otherwise.
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
# The lambda keyword creates a short, one-line anonymous function (a function without a name) that acts as a bridge between pandas and your custom logic.
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
# which does exactly what its name suggests: it splits the dataset into a training set and a test set, 
# and it even shuffles the data for you. 
# Here is how to use it:
import numpy as np
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
# Now that you have created the income category attribute, you can use Scikit-Learn’s StratifiedShuffleSplit class 
# to do a stratified sampling based on this attribute.
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

#### BE CLEAR ABOUT WHY WE ARE USING STRATIFICATION AND RANDOMISATION BOTH READ MORE ABOUT IT

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
# if training set is too large, you can use a smaller subset of the training set to speed up experimentation.
# we have small training set, so we can use the whole training set for experimentation. 
# therefore, we will create a copy of the training set to avoid any side effects on the original data.
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True , s=housing["population"] / 100, label="population",
 c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10, 7))
#The radius of each circle represents the district’s population (option s), and the color represents the
#price (option c). Here you use a predefined color map (option cmap) called jet, which ranges from blue (low values) to red (high prices)
plt.show()
# %%
#for calculating the standard coefficient of correlation (Pearson's r) b/w every pair of attributes, use corr() methhod
corr_matrix= housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
#The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; for example, the median
#house value tends to go up when the median income goes up. When the coefficient is close to –1, it means that there is a strong negative correlation
# %%
# pandas scatter_matrix function plots every numerical attribute against every other numerical attribute.
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))
# %%
#Looking at the correlation scatterplots, it seems like the most promising attribute to predict the median house value is the median income, so you
#zoom in on their scatterplot
housing.plot(kind="scatter", x ="median_income",y= "median_house_value", alpha= 0.1,grid =True,)
plt.show()
# %%
# sometimes already existinng attributes are not sufficient for our machine learning purposes,
# so we add new attributes to fulfuil our purposes. 
housing["rooms_per_household"] =housing["total_rooms"]/housing["households"]
housing["bedrooms_ratio"] =housing["total_bedrooms"]/housing["total_rooms"]
housing["people_per_house"] =housing["population"]/housing["households"]

corr_matrix =housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
#%% 
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "people_per_house", "bedrooms_ratio", "rooms_per_household"]
scatter_matrix(housing[attributes], figsize=(12,8))
# %%
# It’s time to prepare the data for your machine learning algorithms. Instead of doing this manually, you should write functions for this purpose,
# for why we need to do this refer the book.
# first, revert to a clean training set (by copying strat_train_set once again). You should also separate the predictors and the labels, since
# you don’t necessarily want to apply the same transformations to the predictors and the target values.
housing =strat_train_set.drop("median_house_value", axis=1) # drop() - drops the label spefcified in another copy of data it never modify the orginal data
housing_labels = strat_train_set["median_house_value"].copy()

# %%
# sometimes, some attributes may have missing values there are 3 ways to handle this -
# housing.dropna(subset = ["total_bedrooms"]) ---> drops the districts that have no total_bedrooms attribute
# housing.drop("total_bedrooms", axis=1) ---> drops the total_bedrooms attribute from the dataset
# median = housing["total_bedrooms"].median()
# housing = housing.fillna(median, inplace=True) ---> fills the missing values with median value of the total_bedrooms attribute
# the last option is also known as imputation and best of all, scikit-learn provides a handy class to take care of missing values:
from sklearn.impute import  SimpleImputer
imputer = SimpleImputer(strategy ="median")
# Because the imputer calculates statistical averages (like the median), it only works on numerical data.
housing_num = housing.select_dtypes(include=[np.number]) # drops the non numerical attributes
# Next, you "train" the imputer on your numerical dataset using the .fit() command which simply scan all numerical columns andcalculatesthe median for each.
imputer.fit(housing_num)
# The imputer calculates the median value for every single column in your dataset and stores those numbers safely away in a variable called imputer.statistics_
imputer.statistics_
housing_num.median().values
# Once it knows the medians, you use the .transform() command to actually fill in all the missing gaps
X = imputer.transform(housing_num) # output of this will always be an  NumPy array containing neither columns nor rows
# more powerful imputers are - KNNImputer (replaces missing values with k-nearest neighbour) and IterativeImputer(trains a regressive model to pr3edict the missing values iteratively)
# refer book for more details on scikit learn design and API
# we can wrap X in a dataframe easily
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
housing_tr.head()

# %%
# we have dealt with numerical data so far, what if our data contains text attributes? for example, ocean_proximity is a text attribute.
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
# these text are not arbitrary , they are categories, hence ocean_proximity is a categorical attribute.
# since machine learning algorithms prefer to work with numbers, you need to convert these categories from text  to numbers

# %%
# for this we can use OrdinalEncoder class  of scikit learn
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder =OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat) #fit_transform(): Performs both fit() and transform() in a single optimized step on training data
housing_cat_encoded[:10]  # ---> output is a 2D array containing the encoded values of the first 10 districts 
ordinal_encoder.categories_ # ---> output is a list of 1D arrays containing the categories of each attribute
# ordinalencoder assumes that two nearby values are more similar than two distant values, which is not true for categorical attributes. 
# For example, the ocean_proximity attribute has no order between its categories, so it is better to use one-hot encoding instead of ordinal encoding.

#### fit(): Analyzes the dataset and memorizes state. For OneHotEncoder, it notes every unique category present in your training features.
#### transform() uses those learned parameters to convert data values into numerical matrices that an ML algorithm can process.

# %%
#one-hot encoding is a better approach than ordinal encoding for categorical attributes, because it does not assume any order between the categories.
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot =cat_encoder.fit_transform(housing_cat)
housing_cat_1hot # ---> output is a sparse matrix containing the one-hot encoded values of the categorical attribute and it's position instead of numpy array.
housing_cat_1hot.toarray() # ---> output is a dense array containing the one-hot encoded values
# Alternatively, you can set sparse=False when creating the OneHotEncoder, in which case the transform() method will return a regular (dense) NumPy array directly.
# list of categories can be obtained as same as ordinal encoder
cat_encoder.categories_
# %%
# you can also use pandas get_dummies() function to perform one-hot encoding
df_test =pd.DataFrame({"ocean_proximity":["NEAR BAY", "INLAND", "NEAR OCEAN"]})
pd.get_dummies(df_test)
# while get_dummies() is simpler to use, and output is a pandas DataFrame but it is harder to use in a ML pipeline
# and it is mostly used for Quick exploration of the data and visualization.
#%%
# Well, the advantage of OneHotEncoder is that it remembers which categories it was trained on. This is very important because once your
# model is in production, it should be fed exactly the same features as during training: no more, no less.
# while get_dummies() will happily create new columns for any new categories it encounters or ignores any missing categories, which would break your model.
df_test_unknown =pd.DataFrame({"ocean_proximity":["<2H OCEAN", "ISLAND"]})
pd.get_dummies(df_test_unknown)
# this is why get_dummies() is not suitable for production use.
#%%
# OneHotEncoder is smarter: it will detect the unknown category and raise an exception. If you prefer, you can set the handle_unknown
# hyperparameter to "ignore", in which case it will just represent the unknown category with zeros
cat_encoder.handle_unknown ="ignore"
cat_encoder.transform(df_test_unknown).toarray()
# whenever you have a new category, it will be represented as a row of zeros, which is a reasonable approach.
# when the datasets have a large number of categories, the one hot encoding might create a very large number of features, 
# which can be problematic for some machine learning algorithms. In this case, 
# you can use the hashing trick to map categories to integers and then use one-hot encoding on those integers. 
# This is implemented in scikit-learn’s FeatureHasher class.
# When dealing with neural networks, you can replace each category with a low-dimensional vector learned during training, which is called an embedding.
# %%
cat_encoder.feature_names_in_
# When you fit any Scikit-Learn estimator using a DataFrame, the estimator stores the column names in the feature_names_in_ attribute
# Transformers also provide a get_feature_names_out() method that you can use to build a DataFrame around the transformer’s output:
cat_encoder.get_feature_names_out()
# %%
# to improve the bias in algorithms, we can add feature scaling to the data preprocessing pipeline.
# Feature scaling is a technique to standardize the independent features present in the data in a fixed range.
# There are two common ways to do this:
# 1. Min-max scaling (Normalization):  for each attribute, the values are shifted and rescaled so that they end up
# ranging from 0 to 1. This is performed by subtracting the min value and dividing by the difference between the min and the max.
# 2. Standardization: for each attribute, the values are shifted and rescaled so that they end up having a mean of 0 and a standard deviation of 1. 
# This is performed by subtracting the mean value and dividing by the standard deviation.
from sklearn.preprocessing import MinMaxScaler
min_max_scaler =MinMaxScaler(feature_range=(0,1)) #can be set to -1 to 1
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
# %%
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# %%
# When a feature's distribution has a heavy tail, 
# both minmaxing and standardization will squash most values into a small range. 
# ML models don't like them at all because they will be dominated by the few extreme values.
# so before scaling we have to transform the data to make it more symmetrically distributed.
# Another approach to handle heavy-tailed features consists in bucketizing the feature. 
# This means chopping its distribution into roughly equal-sizedbuckets, and replacing 
# each feature value with the index of the bucket it belongs to.
# When a feature has a multimodal distribuition, it is often a good idea to bucketize it and this time
# bucket IDs can be treated as categories rather than as numbers, so you can use one-hot encoding on them.
# This approach will allow the regression model to more easily learn different rules for different ranges of this feature value.
# Another approach to transforming multimodal distributions is to add a feature for each of the modes (at least the main ones),
# The similarity measure is typically computed using a radial basis function(RBF)—
# any function that depends only on the distance between the input value and a fixed point. The most commonly used RBF is the Gaussian RBF
from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
# %%
# in this example we just train the model on a single raw input feature (median income), for simplicity
from sklearn.linear_model import LinearRegression
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# %%
# Sometimes you need to alter your data using a straightforward mathematical calculation (like taking the logarithm of a number). 
# Because this is just a direct calculation, it doesn't require the machine learning model to "train" or learn patterns from the data first.
# let's create a custom log-transformer and apply it to the population feature:
from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(np.log, inverse_func = np.exp)
log_pop = log_transformer.fit_transform(housing[["population"]])
# %%
# here’s a custom transformer that acts much like the StandardScaler:
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
class StandardScalerClone(BaseEstimator, TransformerMixin):
 def __init__(self, with_mean=True): # no *args or **kwargs!
    self.with_mean = with_mean
 def fit(self, X, y=None): # y is required even though we don't use it
    X = check_array(X) # checks that X is an array with finite float values
    self.mean_ = X.mean(axis=0)
    self.scale_ = X.std(axis=0)
    self.n_features_in_ = X.shape[1] # every estimator stores this in fit()
    return self # always return self!
 def transform(self, X):
    check_is_fitted(self) # looks for learned attributes (with trailing _)
    X = check_array(X)
    assert self.n_features_in_ == X.shape[1]
    if self.with_mean:
        X = X - self.mean_
    return X / self.scale_
# %%
# A custom transformer can (and often does) use other estimators in its implementation. 
# For example, the following code demonstrates custom transformer that uses a KMeans clusterer in the fit() method 
# to identify the main clusters in the training data, and then uses rbf_kernel() in the transform() method 
# to measure how similar each sample is to each cluster center:
from sklearn.cluster import KMeans
class ClusterSimilarity(BaseEstimator, TransformerMixin):
 def __init__(self, n_clusters=10, gamma=1.0,random_state=None):
    self.n_clusters = n_clusters
    self.gamma = gamma
    self.random_state = random_state
 def fit(self, X, y=None, sample_weight=None):
    self.kmeans_ = KMeans(self.n_clusters,random_state=self.random_state)
    self.kmeans_.fit(X, sample_weight=sample_weight)
    return self # always return self!
 def transform(self, X):
    return rbf_kernel(X, self.kmeans_.cluster_centers_,gamma=self.gamma)
 def get_feature_names_out(self, names=None):
    return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# %%
