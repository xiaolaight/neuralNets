import pandas
import numpy as np

# In order to do perception, you must know how to manage a dataframe. First, to load in the dataframe:
df = pandas.read_csv(r"C:\Users\andig\Downloads\train.csv\train.csv")

# View your data, and basic information about it
print(df.head()) # first 5 rows
print(df.shape) # dimensions
print(df.isnull().sum()) # gives the amount of null values for each column

# Data can be processed and modified with pandas as well.
# The below method intelligently fills missing values along axis=columns
df.interpolate(method='linear', axis=0)

# You can pull specific rows to look at
print(df.iloc[0])

# You can do it for multiple rows as well
print(df.iloc[[0, 1, 2, 3]])

# If you want to look at specifically columns, you can directly access column names
# In this case, I am using the MNIST digit recognizer dataset. Note that the value placed here is the column index name
print(df["label"])

# You can do the same with this:
print(df.label)

# Convert it into an array with this:
print(df.label.values)

# You can then process the data by removing outliers. This can be done with a simple Z-score method.
# This will not work well with the MNIST digit dataset, but if you were looking at the titanic dataset, you would do:
upper = df["label"].mean() + 3*df["label"].std()
lower = df["label"].mean() - 3*df["label"].std()
good_df = df[(df["label"] < upper) & (df["label"] > lower)]
