#!/bin/bash

cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/
source setenv_pointcept_only.sh
cd /cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept

#checkpoint_file=sonata/lartpc_v5_h200_noghosts/model/epoch_43.pth
#outfile=features_v5_epoch_43_limit_5perparticle.npz
#python3 tools/visualize_sonata_tsne_particle_sampled.py --config configs/lartpc/pretrain-sonata-v1m1-lartpc-v5-p100.py --data-list lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_valsplit.txt --checkpoint ${checkpoint_file} --max-points-per-particle 5 --target-points-per-class 5000 --save-features ${outfile} --stop-after-features


epoch_file=epoch_50
python3 tools/visualize_sonata_tsne_particle_sampled.py --config configs/lartpc/pretrain-sonata-v1m1-lartpc-v6-p100.py --data-list lartpc_data_prep/hdflist_combined_prod4_validated_shuffled_valsplit.txt --checkpoint sonata/lartpc_v6_h200_noghosts_pretrain/model/${epoch_file}.pth --max-points-per-particle 5 --target-points-per-class 5000 --save-features features_v6_${epoch_file}.npz --stop-after-features
