from trendfitter.models.PCA import PCA
import pandas as pd

model = PCA()

data = pd.read_csv('food-texture complete toy_example.csv', index_col=0)

data = (data - data.mean()) / data.std()
print(data.head())
model.fit(data)
print(f' Model fitted with : {model.principal_components} principal components')
print(model.loadings)