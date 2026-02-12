############# Data Pre-processing ##############

################ Type casting #################

# Importing the pandas library for data manipulation and analysis
import pandas as pd  

# Reading data from a CSV file named "ethnic diversity.csv" located at "C:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx")  

# Displaying the data types of each column in the DataFrame
data.dtypes  



# Getting help on the astype method of pandas DataFrame
help(data.astype)

# Converting the 'EmpID' column from 'int64' to 'str' (string) type
data.Record_ID = data.Record_ID.astype('str')

# Displaying the data types after converting 'EmpID' column
data.dtypes



##############################################
### Identify duplicate records in the data ###
# Importing the pandas library for data manipulation and analysis
import pandas as pd  

# Reading data from a CSV file named "mtcars_dup.csv" located at "C:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx")  
# Getting help on the duplicated method of pandas DataFrame
help(data.duplicated)

# Finding duplicate rows in the DataFrame and storing the result in a Boolean Series
duplicate = data.duplicated()  # Returns Boolean Series denoting duplicate rows.

# Displaying the Boolean Series indicating duplicate rows
duplicate

# Counting the total number of duplicate rows
sum(duplicate)

# Finding duplicate rows in the DataFrame and keeping the last occurrence of each duplicated row
duplicate = data.duplicated(keep = 'last')
duplicate

# Finding all duplicate rows in the DataFrame
duplicate = data.duplicated(keep = False)
duplicate

# Removing duplicate rows from the DataFrame and storing the result in a new DataFrame
data1 = data.drop_duplicates() # Returns DataFrame with duplicate rows removed.

# Removing duplicate rows from the DataFrame and keeping the last occurrence of each duplicated row
data1 = data.drop_duplicates(keep = 'last')

# Removing all duplicate rows from the DataFrame
data1 = data.drop_duplicates(keep = False)


# Duplicates in Columns
# We can use correlation coefficient values to identify columns which have duplicate information

# Importing the pandas library for data manipulation and analysis
import pandas as pd  

# Reading data from a CSV file named "Cars.csv" located at "C:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx")  

# Correlation coefficient
'''
Ranges from -1 to +1. 
Rule of thumb says |r| > 0.85 is a strong relation
'''
# Calculating the correlation matrix for the columns in the DataFrame
data1.corr(numeric_only=True)

'''We can observe that none of the features show strong linear correlation with each other, as all correlation coefficients are close to zero. 
This indicates low multicollinearity, so no feature needs to be removed based on correlation analysis.
'''


################################################
############## Outlier Treatment ###############
# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

# Reading data from a CSV file named "ethnic diversity.csv" located at "C:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx")  

# Displaying the data types of each column in the DataFrame
data.dtypes

# Creating a box plot to visualize the distribution and potential outliers in the 'Salaries' column
num_cols = [
    'Sperm_Concentration_M_per_ml'
]

sns.boxplot(data=data[num_cols])
plt.xticks(rotation=45)
plt.show()
# Creating a box plot to visualize the distribution and potential outliers in the 'age' column
num_cols = [
    'Total_Motility_Percent'
]

sns.boxplot(data=data[num_cols])
plt.xticks(rotation=45)
plt.show()
# No outliers in age column

# Detection of outliers in the 'Salaries' column using the Interquartile Range (IQR) method
col = 'Sperm_Concentration_M_per_ml'

Q1 = data[col].quantile(0.25)
Q3 = data[col].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR


############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# Let's flag the outliers in the dataset
col = 'Sperm_Concentration_M_per_ml'
# Creating a boolean array indicating whether each value in the 'Salaries' column is an outlier
outliers_df = np.where(
    data[col] > upper_limit, True,
    np.where(data[col] < lower_limit, True, False)
)
# Filtering the DataFrame to include only rows where 'Salaries' column contains outliers
df_out = data.loc[outliers_df]

# Filtering the DataFrame to exclude rows containing outliers
df_trimmed = data.loc[~outliers_df]

# Displaying the shape of the original DataFrame and the trimmed DataFrame
data.shape, df_trimmed.shape

# Creating a box plot to visualize the distribution of 'Salaries' in the trimmed dataset
sns.boxplot(df_trimmed[col])

############### 2. Replace ###############
# Replace the outliers by the maximum and minimum limit
# Creating a new column 'df_replaced' in the DataFrame with values replaced by upper or lower limit if they are outliers
data[col + '_replaced'] = np.where(
    data[col] > upper_limit, upper_limit,
    np.where(data[col] < lower_limit, lower_limit, data[col])
)

# Creating a box plot to visualize the distribution of 'df_replaced' column
sns.boxplot(data[col + '_replaced'])


############### 3. Winsorization ###############
# pip install feature_engine   # install the package
# Importing the Winsorizer class from the feature_engine.outliers module
from feature_engine.outliers import Winsorizer

# Defining the Winsorizer model with IQR method
# Parameters:
# - capping_method: 'iqr' specifies the Interquartile Range (IQR) method for capping outliers
# - tail: 'both' indicates that both tails of the distribution will be capped
# - fold: 1.5 specifies the multiplier to determine the range for capping outliers based on IQR
# - variables: ['Salaries'] specifies the column(s) in the DataFrame to apply the Winsorizer to
col = 'Sperm_Concentration_M_per_ml'

winsor_iqr = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=[col]
)


# Fitting the Winsorizer model to the 'Salaries' column and transforming the data
df_s = winsor_iqr.fit_transform(data[[col]])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Creating a box plot to visualize the distribution of 'Salaries' after applying Winsorizer with IQR method
sns.boxplot(df_s[col])

# Defining the Winsorizer model with Gaussian method
# Parameters:
# - capping_method: 'gaussian' specifies the Gaussian method for capping outliers
# - tail: 'both' indicates that both tails of the distribution will be capped
# - fold: 3 specifies the number of standard deviations to determine the range for capping outliers based on Gaussian method
# - variables: ['Salaries'] specifies the column(s) in the DataFrame to apply the Winsorizer to
winsor_gaussian = Winsorizer(
    capping_method='gaussian',
    tail='both',
    fold=3,
    variables=[col]
)
# Fitting the Winsorizer model to the 'Salaries' column and transforming the data
df_t = winsor_gaussian.fit_transform(data[[col]])

# Creating a box plot to visualize the distribution of 'Salaries' after applying Winsorizer with Gaussian method
sns.boxplot(df_t[col])


# Define the model with percentiles:
# Default values
# Right tail: 95th percentile
# Left tail: 5th percentile

# Defining the Winsorizer model with quantiles method
# Parameters:
# - capping_method: 'quantiles' specifies the quantiles method for capping outliers
# - tail: 'both' indicates that both tails of the distribution will be capped
# - fold: 0.05 specifies the proportion of data to be excluded from the lower and upper ends of the distribution (5th and 95th percentiles)
# - variables: ['Salaries'] specifies the column(s) in the DataFrame to apply the Winsorizer to

winsor_percentile = Winsorizer(
    capping_method='quantiles',
    tail='both',
    fold=0.05,
    variables=[col]
)

# Fitting the Winsorizer model to the 'Salaries' column and transforming the data
df_p = winsor_percentile.fit_transform(data[[col]])

# Creating a box plot to visualize the distribution of 'Salaries' after applying Winsorizer with quantiles method
sns.boxplot(df_p[col])


##############################################
#### zero variance and near zero variance ####

# Importing the pandas library for data manipulation and analysis
import pandas as pd  

# Reading data from a CSV file named "ethnic diversity.csv" located at "C:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx")  

# Displaying the data types of each column in the DataFrame
data.dtypes
# If the variance is low or close to zero, then a feature is approximately constant and will not improve the performance of the model.
# In that case, it should be removed. 

# Select only numeric columns
numeric_columns = data.select_dtypes(include = np.number)

# Calculating the variance of each numeric variable in the DataFrame
numeric_columns.var()

# Checking if the variance of each numeric variable is equal to 0 and returning a boolean Series
numeric_columns.var() == 0 

# Checking if the variance of each numeric variable along axis 0 (columns) is equal to 0 and returning a boolean Series
numeric_columns.var(axis = 0) == 0 


#############
# Discretization

# Importing the pandas library for data manipulation and analysis
import pandas as pd

# Reading data from a CSV file named "ethnic diversity.csv" located at "C:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx")  

# Displaying the first few rows of the DataFrame
data.head()

# Displaying the last few rows of the DataFrame
data.tail()

# Displaying information about the DataFrame, including the data types of each column and memory usage
data.info()

# Generating descriptive statistics of the DataFrame, including count, mean, standard deviation, minimum, maximum, and quartile values
data.describe()

# Binarizing the 'Salaries' column into two categories ('Low' and 'High') based on custom bins
data['Sperm_Concentration_Level'] = pd.qcut(
    data['Sperm_Concentration_M_per_ml'],
    q=2,
    labels=["Low", "High"]
)

# Counting the number of occurrences of each category in the 'Salaries_new' column
data.Sperm_Concentration_Level.value_counts()


''' We can observe that the total number of values are 309. This is because one of the value has become NA.
This happens as the cut function by default does not consider the lowest (min) value while discretizing the values.
To over come this issue we can use the parameter 'include_lowest' set to True.
'''

# Binarizing the 'Salaries' column into two categories ('Low' and 'High') based on custom bins
# Parameters:
# - bins: Custom bins defined by the minimum salary value, mean salary value, and maximum salary value
# - include_lowest: Whether to include the lowest edge of the bins in the intervals
# - labels: Labels assigned to the resulting categories
data['Sperm_Concentration_Level'] = pd.cut(
    data['Sperm_Concentration_M_per_ml'],
    bins=[data['Sperm_Concentration_M_per_ml'].min(),
          data['Sperm_Concentration_M_per_ml'].mean(),
          data['Sperm_Concentration_M_per_ml'].max()],
    labels=["Low", "High"]
)

# Counting the number of occurrences of each category in the 'Salaries_new1' column
data.Sperm_Concentration_Level.value_counts()

#########
# Importing the matplotlib library for creating plots
import matplotlib.pyplot as plt

# Creating a bar plot to visualize the distribution of 'Salaries_new1' categories
data['Sperm_Concentration_Level'].value_counts().plot(kind='bar')
plt.title("Sperm Concentration Levels")
plt.show()


# Creating a histogram to visualize the distribution of 'Salaries_new1' categories
plt.hist(data['Sperm_Concentration_M_per_ml'], bins=20)
plt.title("Distribution of Sperm Concentration")
plt.show()

# Creating a box plot to visualize the distribution of 'Salaries_new1' categories
plt.boxplot(data['Sperm_Concentration_M_per_ml'])
plt.title("Outliers in Sperm Concentration")
plt.show()

# Discretization into multiple bins based on quartiles
data['Sperm_Concentration_Group'] = pd.cut(
    data['Sperm_Concentration_M_per_ml'],
    bins=[
        data['Sperm_Concentration_M_per_ml'].min(),
        data['Sperm_Concentration_M_per_ml'].quantile(0.25),
        data['Sperm_Concentration_M_per_ml'].quantile(0.50),
        data['Sperm_Concentration_M_per_ml'].quantile(0.75),
        data['Sperm_Concentration_M_per_ml'].max()
    ],
    labels=["Low", "Medium", "High", "Very High"]
)

# Counting the number of occurrences of each category in the 'Salaries_multi' column
data['Sperm_Concentration_Group'].value_counts()

# Counting the number of occurrences of each category in the 'MaritalDesc' column
data['Usable_Embryo'].value_counts(normalize=True)


##################################################
################## Dummy Variables ###############
# methods:
    # get dummies
    # One Hot Encoding
    # Label Encoding
    # Ordinal Encoding
# Importing the pandas library for data manipulation and analysis
import pandas as pd
# Importing the numpy library for numerical computing
import numpy as np

# Reading data from a CSV file named "ethnic diversity.csv" located at "C:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx")  


# Displaying the names of all columns in the DataFrame
data.columns 

# Displaying the shape of the DataFrame (number of rows and columns)
data.shape 

# Displaying the data types of each column in the DataFrame
data.dtypes

# Displaying concise summary of the DataFrame including non-null counts and data types
data.info()

# 1. Dropping ID-related columns and storing in a new DataFrame
df1 = data.drop(['Record_ID', 'Patient_ID', 'Oocyte_ID'], axis = 1)

# 2. Dropping ID-related columns from the DataFrame inplace
data.drop(['Record_ID', 'Patient_ID', 'Oocyte_ID'], axis = 1, inplace = True)

# 3. Creating dummy variables and converting Boolean values to integers (1s and 0s)
df_new = pd.get_dummies(data1).astype('object')

# 4. Creating dummy variables and dropping the first category of each column
df_new_1 = pd.get_dummies(data, drop_first = True).astype('object')

# 5. Displaying the names of all columns remaining in the DataFrame
print(data.columns)

# 6. Selecting specific clinical columns and updating the DataFrame
df = data[['Sperm_Concentration_M_per_ml', 'Total_Motility_Percent', 'Head_Shape_Score', 
         'Motility_Pattern', 'Fertilization_Success', 'Usable_Embryo', 'Microscope_Type']]

# 7. Extracting 'Sperm_Concentration_M_per_ml' as a pandas Series
a = df['Sperm_Concentration_M_per_ml']

# 8. Extracting 'Sperm_Concentration_M_per_ml' as a DataFrame
b = df[['Sperm_Concentration_M_per_ml']]



# Importing the OneHotEncoder class from the sklearn.preprocessing module
from sklearn.preprocessing import OneHotEncoder
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx")

# Creating an instance of the OneHotEncoder
enc = OneHotEncoder(sparse_output = False) # initializing method 
# setting sparse_output=False explicitly instructs the OneHotEncoder to return a dense array instead of a sparse matrix.

# Transforming the categorical columns (from Position column onwards) into one-hot encoded format and converting to DataFrame
categorical_cols = ['Head_Shape_Score',  'Motility_Pattern', 'Microscope_Type']
enc_df = pd.DataFrame(enc.fit_transform(data[categorical_cols]), 
                      columns=enc.get_feature_names_out(input_features=categorical_cols))

#######################
# Label Encoder
# Label Encoding is typically applied to a single column or feature at a time, meaning it operates on one-dimensional data.
# Importing the LabelEncoder class from the sklearn.preprocessing module
from sklearn.preprocessing import LabelEncoder

# Creating an instance of the LabelEncoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
# X contains the features (independent variables), excluding the last column
X = data.iloc[:, :9].copy()
# y contains the target variable (dependent variable), which is the last column
y = data.iloc[:, 9]

# Transforming the 'Sex' column into numerical labels using LabelEncoder
X['Selection_Date'] = LabelEncoder().fit_transform(X['Selection_Date'])
X['Embryologist_ID'] = LabelEncoder().fit_transform(X['Embryologist_ID'])
X['Cycle_Number'] = LabelEncoder().fit_transform(X['Cycle_Number'].astype(str))
########################
# Ordinal Encoding
# Importing the OrdinalEncoder class from the sklearn.preprocessing module
from sklearn.preprocessing import OrdinalEncoder
# Ordinal Encoding can handle multiple dimensions or features simultaneously.
oe = OrdinalEncoder()
# Data Split into Input and Output variables
# X contains the features (independent variables), excluding the last column
X = data.iloc[:, :9]
# y contains the target variable (dependent variable), which is the last column
y = data.iloc[:, 9]

target_cols = ['Selection_Date', 'Cycle_Number', 'Embryologist_ID']
X[target_cols] = oe.fit_transform(X[target_cols])


#################### Missing Values - Imputation ###########################
# Importing the necessary libraries
import numpy as np
import pandas as pd

# Loading the modified ethnic dataset from a CSV file located at "C:/Data/modified ethnic.csv"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx")  # for doing modifications

# Checking for the count of missing values (NA's) in each column of the DataFrame
df.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data (Salaries)
# Mode is used for discrete data (ex: Position, Sex, MaritalDesc)

# Importing SimpleImputer from the sklearn.impute module
from sklearn.impute import SimpleImputer

# Mean Imputer: Replacing missing values in the 'Salaries' column with the mean value
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

data["Sperm_Concentration_M_per_ml"] = mean_imputer.fit_transform(
    data[["Sperm_Concentration_M_per_ml"]]
)

data["Total_Motility_Percent"] = mean_imputer.fit_transform(
    data[["Total_Motility_Percent"]]
)

data["Sperm_Concentration_M_per_ml"].isna().sum()
data["Total_Motility_Percent"].isna().sum()

# Median Imputer: Replacing missing values in the 'age' column with the median value
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')

data["Selection_Time_Seconds"] = median_imputer.fit_transform(
    data[["Selection_Time_Seconds"]]
)

data["Selection_Time_Seconds"].isna().sum() 

# Mode Imputer: Replacing missing values in the 'Sex' and 'MaritalDesc' columns with the most frequent value
mode_imputer = SimpleImputer(strategy='most_frequent')
data[['Motility_Pattern', 'Vacuoles_Present']] = mode_imputer.fit_transform(
    data[['Motility_Pattern', 'Vacuoles_Present']]
)

# Constant Value Imputer: Replacing missing values in the 'Sex' column with a constant value 'F'
constant_imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='constant',
    fill_value="Normal"
)

data["Acrosome_Status"] = constant_imputer.fit_transform(
    data[["Acrosome_Status"]]
).ravel()

data["Acrosome_Status"].isna().sum()


# Random Imputer: Replacing missing values in the 'age' column with random samples from the same column
from feature_engine.imputation import RandomSampleImputer

random_imputer = RandomSampleImputer(
    variables=["Embryologist_Experience_Years"]
)

data["Embryologist_Experience_Years"] = random_imputer.fit_transform(
    data[["Embryologist_Experience_Years"]]
)

data["Embryologist_Experience_Years"].isna().sum()
data.isnull().sum()


#####################
# Normal Quantile-Quantile Plot

# Importing pandas library for data manipulation
import pandas as pd

# Reading data from a CSV file named "education.csv" located at "C:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx") 

# Importing scipy.stats module for statistical functions
import scipy.stats as stats
# Importing pylab module for creating plots
import pylab

# Checking whether the 'gmat' data is normally distributed using a Q-Q plot
stats.probplot(data["Sperm_Concentration_M_per_ml"], dist="norm", plot=plt)
plt.title("Q-Q plot for Sperm Concentration")
plt.show()

# Importing numpy module for numerical computations
import numpy as np

# Transformation to make 'workex' variable normal by applying logarithmic transformation
stats.probplot(
    np.log(data["Sperm_Concentration_M_per_ml"]),
    dist="norm",
    plot=pylab
)
pylab.show()
# Importing seaborn and matplotlib.pyplot for plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Original data
prob = stats.probplot(
    data["Sperm_Concentration_M_per_ml"],
    dist=stats.norm,
    plot=pylab
)
# Transforming the 'workex' data using Box-Cox transformation and saving the lambda value
fitted_data, fitted_lambda = stats.boxcox(data["Sperm_Concentration_M_per_ml"])

# Creating subplots

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plotting the original and transformed data distributions
sns.kdeplot(
    data["Selection_Time_Seconds"],
    fill=True,            # replaces shade=True
    linewidth=2,
    color="green",
    label="Non-Normal",
    ax=ax[0]
)
ax[0].set_title("Before Transformation")
ax[0].legend(loc="upper right")

# Box-Cox transformed data distribution
sns.kdeplot(
    fitted_data,
    fill=True,            # replaces shade=True
    linewidth=2,
    color="blue",
    label="Normal",
    ax=ax[1]
)
ax[1].set_title("After Box-Cox Transformation")
ax[1].legend(loc="upper right")

plt.show()

# Adding legends to the subplots
plt.legend(loc = "upper right")

# Rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)

# Printing the lambda value used for transformation
print(f"Lambda value used for Transformation: {fitted_lambda}")

# Transformed data
prob = stats.probplot(fitted_data, dist=stats.norm, plot = pylab)


# Yeo-Johnson Transform

'''
We can apply it to our dataset without scaling the data.
It supports zero values and negative values. It does not require the values for 
each input variable to be strictly positive. 

In Box-Cox transform the input variable has to be positive.
'''

# Importing pandas library for data manipulation
import pandas as pd
# Importing stats module from scipy library for statistical functions
from scipy import stats

# Importing seaborn and matplotlib.pyplot for plotting
import seaborn as sns
import matplotlib.pyplot as plt
# Importing pylab module for creating plots
import pylab

# Read data from a CSV file named "education.csv" located at "C:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx") 

# Original data
# Checking whether the 'workex' data is normally distributed using a Q-Q plot
prob = stats.probplot(
    data["Selection_Time_Seconds"],
    dist=stats.norm,
    plot=pylab
)


# Importing transformation module from feature_engine library
from feature_engine import transformation

# Set up the Yeo-Johnson transformer for 'workex' variable
tf = transformation.YeoJohnsonTransformer(
    variables=["Selection_Time_Seconds"]
)

df_tf = tf.fit_transform(data)


# Transformed data
# Checking whether the transformed 'workex' data is normally distributed using a Q-Q plot
prob = stats.probplot(
    df_tf["Selection_Time_Seconds"],
    dist=stats.norm,
    plot=pylab
)



####################################################
######## Standardization and Normalization #########

# Importing pandas library for data manipulation
import pandas as pd
# Importing numpy library for numerical computations
import numpy as np

# Reading data from a CSV file named "mtcars.csv" located at "D:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx") 


# Generating descriptive statistics of the original data
a = data.describe()

# Importing StandardScaler from the sklearn.preprocessing module
from sklearn.preprocessing import StandardScaler

# Initialise the StandardScaler
scaler = StandardScaler()

# Scaling the data using StandardScaler
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()

data_scaled = scaler.fit_transform(data[numeric_cols])

# Converting the scaled array back to a DataFrame
dataset = pd.DataFrame(df)

# Generating descriptive statistics of the scaled data
res = dataset.describe()


# Normalization
''' Alternatively we can use the below function'''
# Importing MinMaxScaler from the sklearn.preprocessing module
from sklearn.preprocessing import MinMaxScaler

# Initializing the MinMaxScaler
minmaxscale = MinMaxScaler()

# Scaling the data using MinMaxScaler
# Select numeric columns only
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Fit and transform
df_n = minmaxscale.fit_transform(data[numeric_cols])

# Optional: convert back to DataFrame with column names
df_n = pd.DataFrame(df_n, columns=numeric_cols)

# Converting the scaled array back to a DataFrame
dataset1 = pd.DataFrame(df_n)

# Generating descriptive statistics of the scaled data
res1 = dataset1.describe()


### Normalization
# Load dataset from a CSV file named "ethnic diversity.csv" located at "D:/Data/"
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx") 
# Displaying column names of the dataset
data.columns

# Dropping columns 'Employee_Name', 'EmpID', 'Zip' from the dataset
data.columns
cols_to_drop = [
    "Patient_ID",
    "Sample_ID",
    "Embryologist_Name"
]

# Generating descriptive statistics of the original dataset
a1 = data.describe()

# Generating dummy variables for categorical columns in the dataset and dropping the first category of each column

# Selecting a categorical variable (similar to ethnic1)
sperm_cat = df['Motility_Pattern']

# Creating dummy variables
sperm_dummies = pd.get_dummies(sperm_cat, drop_first=True).astype(int)

# Descriptive statistics for dummy variables
a2 = sperm_dummies.describe()

# -------------------------------
# Normalization function
# -------------------------------
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

# Applying normalization
df_norm = norm_func(sperm_dummies)

# Descriptive statistics after normalization
b = df_norm.describe()
sperm_cat = data['Motility_Pattern']
sperm_cat = data['Microscope_Type']
# or

# or
sperm_cat = data['Acrosome_Status']
cat_cols = ['Motility_Pattern', 'Microscope_Type']

sperm_dummies = pd.get_dummies(df[cat_cols], drop_first=True).astype(int)

a2 = sperm_dummies.describe()
df_norm = norm_func(sperm_dummies)
b = df_norm.describe()

# Selecting a categorical column (example)
sperm_cat = df['Motility_Pattern']

# Creating dummy variables
sperm_dummies = pd.get_dummies(sperm_cat, drop_first=True).astype(int)

# Initializing the MinMaxScaler
minmaxscale = MinMaxScaler()

# Scaling the dataset using MinMaxScaler
sperm_minmax = minmaxscale.fit_transform(sperm_dummies)

# Converting the scaled array back to a DataFrame
df_sperm_minmax = pd.DataFrame(
    sperm_minmax,
    columns=sperm_dummies.columns
)

# Generating descriptive statistics after Min-Max scaling
minmax_res = df_sperm_minmax.describe()

import pandas as pd
from sklearn.preprocessing import RobustScaler

# Load the sperm dataset
data = pd.read_excel(r"D:\pranav\PROJECT\Data Set\Sperm_Selection_Dataset_DA.xlsx") 

# Selecting a categorical column (example)
sperm_cat = df['Motility_Pattern']

# Creating dummy variables
sperm_dummies = pd.get_dummies(sperm_cat, drop_first=True).astype(int)

# Initializing the RobustScaler
robust_model = RobustScaler()

# Scaling the dataset using RobustScaler
df_robust = robust_model.fit_transform(sperm_dummies)

# Converting the scaled array back to a DataFrame
dataset_robust = pd.DataFrame(
    df_robust,
    columns=sperm_dummies.columns
)

# Generating descriptive statistics after Robust scaling
res_robust = dataset_robust.describe()
