import os,sys
import pointcept

from pointcept.datasets import LArTPCDataset, LArTPCInstanceDataset

x = LArTPCDataset(coord_scale=0.001,
                  #data_root="data/lartpc",
                  data_list_file="/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept/train_split.txt",
                  exclude_other=True,
                  include_ghosts=True)

data = x.get_data(0)
print(data)
