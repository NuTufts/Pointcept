import os,sys
import pointcept

from pointcept.datasets import LArTPCDataset, LArTPCInstanceDataset

x = LArTPCDataset(data_root="data/lartpc", coord_scale=0.001)
print(x)

data = x.get_data(0)
print(data)
