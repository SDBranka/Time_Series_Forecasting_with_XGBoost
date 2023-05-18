import pandas as pd
import numpy as np


# Dataframes are a very essential concept in Python and filtration of data is 
# required can be performed based on various conditions. They can be achieved 
# in any one of the above ways. Points to be noted:
# - loc works with column labels and indexes.
# - eval and query works only with columns.
# - Boolean indexing works with values in a column only.


dataFrame = pd.DataFrame({'Name': [' RACHEL  ', ' MONICA  ', ' PHOEBE  ',
                                    '  ROSS    ', 'CHANDLER', ' JOEY    '],
                            'Age': [30, 35, 37, 33, 34, 30],
                            'Salary': [100000, 93000, 88000, 120000, 94000, 95000],
                            'JOB': ['DESIGNER', 'CHEF', 'MASUS', 'PALENTOLOGY',
                                    'IT', 'ARTIST']})

# print(dataFrame)
#          Name  Age  Salary          JOB
# 0    RACHEL     30  100000     DESIGNER
# 1    MONICA     35   93000         CHEF
# 2    PHOEBE     37   88000        MASUS
# 3    ROSS       33  120000  PALENTOLOGY
# 4    CHANDLER   34   94000           IT
# 5    JOEY       30   95000       ARTIST


# Method 1: Using loc
# Here we will get all rows having Salary greater or equal to 100000 and 
# Age < 40 and their JOB starts with ‘D’ from the dataframe. Print the 
# details with Name and their JOB. For the above requirement, we can achieve
# this by using loc. It is used to access single or more rows and columns by 
# label(s) or by a boolean array. loc works with column labels and indexes.
filter1 = dataFrame.loc[(dataFrame['Salary']>=100000) & (dataFrame['Age']< 40) & (dataFrame['JOB'].str.startswith('D')),
                    ['Name','JOB']]
# print(filter1)
#         Name       JOB
# 0   RACHEL    DESIGNER
# Output resolves for the given conditions and shows only 2 columns: Name and JOB.


# Method 2: Using NumPy
# Here will get all rows having Salary greater or equal to 100000 and 
# Age < 40 and their JOB starts with ‘D’ from the data frame. We need to 
# use NumPy. 

# filter dataframe                                  
filter2 = np.where((dataFrame['Salary']>=100000) & (dataFrame['Age']< 40) & (dataFrame['JOB'].str.startswith('D')))
# print(f"filter2: \n{filter2}")
# (array([0], dtype=int64),)
# print(f"filtered values:\n{dataFrame.loc[filter2]}")
# index of filtered values:        
# Name  Age  Salary       JOB
# 0   RACHEL     30  100000  DESIGNER

# In the above example, print(filter2) will give the output as 
# (array([0], dtype=int64),)  which indicates the first row with index value 
# 0 will be the output. After that output will have 1 row with all the 
# columns and it is retrieved as per the given conditions.


# Method 3: Using Query (eval and query works only with columns)
# In this approach, we get all rows having Salary lesser or equal to 100000 
# and Age < 40, and their JOB starts with ‘C’ from the dataframe. Its just 
# query the columns of a DataFrame with a single or more Boolean expressions 
# and if multiple, it is having & condition in the middle.
# filter dataframe
filter3 = dataFrame.query('Salary  <= 100000 & Age < 40 & JOB.str.startswith("C").values')
# print(filter3)
#         Name  Age  Salary   JOB
# 1   MONICA     35   93000  CHEF


# Method 4: pandas Boolean indexing multiple conditions standard way 
# (“Boolean indexing” works with values in a column only)
# In this approach, we get all rows having Salary lesser or equal to 100000 
# and Age < 40 and their JOB starts with ‘P’ from the dataframe. In order to 
# select the subset of data using the values in the dataframe and applying 
# Boolean conditions, we need to follow these ways
filter4 = dataFrame[(dataFrame['Salary']>=100000) & (dataFrame['Age']<40) & dataFrame['JOB'].str.startswith('P')][['Name','Age','Salary']]
# print(filter4)
#          Name  Age  Salary
# 3    ROSS       33  120000

# We are mentioning a list of columns that need to be retrieved along with 
# the Boolean conditions and since many conditions, it is having ‘&’.

# Method 5: Eval multiple conditions  (“eval” and “query” works only with 
# columns )
# Here, we get all rows having Salary lesser or equal to 100000 and Age < 40 
# and their JOB starts with ‘A’ from the dataframe. 
filter5 = dataFrame[dataFrame.eval("Salary <=100000 & (Age <40) & JOB.str.startswith('A').values")]
# print(filter5)
#         Name  Age  Salary     JOB
# 5   JOEY       30   95000  ARTIST











