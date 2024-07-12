import pandas as pd
import numpy as np 
import re
data = pd.read_csv("BL MAKER.csv")
irrelevant_columns = ["Edition Statement", "Corporate Author", "Corporate Contributors", "Former owner", "Engraver", "Issuance type", "Shelfmarks"]
data.drop(columns=irrelevant_columns, inplace=True)
data.set_index('Identifier', inplace=True)
data['Date of Publication'] = data['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
data['Place of Publication'] = np.where(data['Place of Publication'].str.contains('London'), 'London', data['Place of Publication'])
data['Place of Publication'] = np.where(data['Place of Publication'].str.contains('Oxford'), 'Oxford', data['Place of Publication'])
print(data.head())