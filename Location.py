import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"datasets/datafile.csv")
df.head()

top_locations = df['location'].value_counts().to_dict()
top_cities = df['location'].value_counts()
print(top_locations)
print(type(top_locations))

x_values = list(top_locations.keys())
y_values = list(top_locations.values())

print(x_values, y_values)

plt.figure(figsize=(8,5))
sns.barplot(x=x_values, y=y_values, palette='mako')
plt.xlabel("Cities")
plt.ylabel("Count")
plt.title("Count of users from various cities in India", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
top_cities.plot(kind="bar", color = "green")
plt.xlabel("Cities")
plt.ylabel("Count")
plt.title("Count of users from various cities in India", fontsize=16)
plt.tight_layout()
plt.show()