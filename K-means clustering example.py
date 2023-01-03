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
