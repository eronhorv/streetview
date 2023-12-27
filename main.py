import pandas as pd
from PIL import Image
import io
import numpy as np
import pickle
import matplotlib.pyplot as plt

LOAD = False

df = pd.read_parquet('data/data1.parquet', engine='fastparquet')

rows = df.shape[0]

cntry = df.iloc[:, 0].values
cntry_sorted = np.sort(np.unique(cntry))

im_bytes = df.iloc[:, 4].values
im_shape = np.array(Image.open(io.BytesIO(im_bytes[0]))).shape

if LOAD:
    base_height = 128
    hpercent = (base_height / float(im_shape[0]))
    wsize = int((im_shape[1] * float(hpercent)))

    ims = np.zeros((rows, base_height, wsize, 3))
    for i, im in enumerate(im_bytes):
       ims[i] = np.array(Image.open(io.BytesIO(im)).resize((wsize, base_height), Image.Resampling.LANCZOS))
    
    np.save('data.npy', ims)
else:
    ims =  np.load('data.npy')

rng = np.random.default_rng()
perm = rng.permutation(rows)

im = ims[5]
im = im / 256

plt.imshow(im)
plt.show()


"""
for i in range(df.shape[0]):
    image_data = df.iloc[i]['image.bytes']
    image = Image.open(io.BytesIO(image_data))

    nparr = np.array(Image.open(io.BytesIO(image_data)))

    #plt.imshow(nparr)
    #plt.show()
    if i % 100 == 0:
        print(nparr.shape)
        print(df.iloc[i]['country_iso_alpha2'])

        """