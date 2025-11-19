from os import walk
import numpy as np


dirA = 'C:/Users/hady1/PycharmProjects/trans_gan_project/datasets/L1000_4Data/RAND_50/trainMCF7/'
dirB = 'C:/Users/hady1/PycharmProjects/trans_gan_project/datasets/L1000_4Data/RAND_50/trainHELA/'

#dirN = 'C:/Users/hady1/PycharmProjects/trans_gan_project/datasets/L1000_4Data/negative/'

filenames = next(walk(dirA), (None, None, []))[2]
negativs = []
for file in filenames:
    #load A
    sampleA = np.genfromtxt(dirA + file, dtype=float)

    #load B
    sampleB = np.genfromtxt(dirB + file, dtype=float)

    #get |A-B|
    neg_control = np.absolute(np.subtract(sampleA,sampleB))
    negativs.append(neg_control)
    #save negative control
    # file = open(dirN + file, 'w')
    # for i in neg_control:
    #     file.write(str(i) + "\t")
    # file.close()