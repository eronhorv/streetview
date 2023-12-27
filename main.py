import pandas as pd
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
#
frame = pd.read_parquet('data/data1.parquet', engine='fastparquet')

image_data = frame.iloc[911]['image.bytes']
image = Image.open(io.BytesIO(image_data))

nparr = np.array(Image.open(io.BytesIO(image_data)))
plt.imshow(nparr)
plt.show()