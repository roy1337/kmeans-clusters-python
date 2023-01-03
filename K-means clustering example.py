## Here is an example of how to  use K-Means clustering to group customers into clusters based on their age, income, and spending habits:

from sklearn.cluster import KMeans
import pandas as pd

# load customer data
customer_data = pd.read_csv("customer_data.csv")

# select relevant columns
data = customer_data[["age", "income", "spending"]]

# create kmeans model
kmeans = KMeans(n_clusters=4)

# fit and predict clusters
predictions = kmeans.fit_predict(data)

# add cluster predictions to dataframe
customer_data["cluster"] = predictions

# print cluster counts
print(customer_data["cluster"].value_counts())





## Output example 

# customer data
   customer_id   age  income  spending  cluster
0             1  37.0  75000.0      20.0        1
1             2  45.0  50000.0      30.0        3
2             3  63.0  80000.0      40.0        2
3             4  25.0  45000.0      35.0        0
4             5  35.0  60000.0      25.0        1

# cluster counts
1    2
3    1
2    1
0    1
Name: cluster, dtype: int64


## This shows the original customer data with an added column for the cluster predictions. The cluster counts show that there are 2 customers in cluster 1, 1 customer in cluster 3, 1 customer in cluster 2, and 1 customer in cluster 0.
## Keep in mind that this is just an example and the actual data and cluster assignments will depend on the specifics of your dataset and the random initialization of the KMeans model.
