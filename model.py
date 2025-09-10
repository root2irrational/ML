

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Set the path to the file you'd like to load
MINUTE_DATA = "eurusd_minute.csv"
HOUR_DATA = "eurusd_hour.csv"
MAX_ROWS = 10000
file_path = HOUR_DATA
# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "imetomi/eur-usd-forex-pair-historical-data-2002-2019",
  file_path,
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

df = pd.DataFrame(df)
df = df[:MAX_ROWS]
rows, cols = df.shape
trainRows = np.floor(rows * 0.8)
testRows = np.ceil(rows * 0.2)
print(f"80/20 split: {int(trainRows)} / {int(testRows)} ")

fig, ax = plt.subplots(1, 1)
ax.table(cellText=df.values, colLabels=df.keys(), loc='center')
plt.show()