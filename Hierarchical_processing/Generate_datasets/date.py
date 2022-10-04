import h5py
import numpy as np
imgData = np.zeros((30,3,128,256))
f = h5py.File('date.h5','w')
f['date'] = imgData
f['labels'] = range(100)
f.close()