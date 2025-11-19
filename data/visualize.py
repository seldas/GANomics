import hypertools as hyp
import numpy as np
import scipy
import pandas as pd
from scipy.linalg import toeplitz
from copy import copy


prediction = 'C:/Users/hady1/PycharmProjects/trans_gan_project/datasets/NB_Data/trainAG/0.txt'

df = pd.read_csv(prediction, delimiter='\t', dtype=float, header=None)
y = df.to_numpy()[:,:10041]
x = np.random.random([10041, 2])
geo = hyp.plot(x)

x = 0