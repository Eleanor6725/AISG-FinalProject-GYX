#!/bin/bash
# chmod +x run.sh
seed=516
common="--seed $seed --no_cuda"

python exp-mnist.py $common --trainer ERM > ERM_log.txt
python exp-mnist.py $common --reg_name IRM --trainer ERM > IRM_log.txt
python exp-mnist.py $common --trainer groupDRO > groupDRO_log.txt
python exp-mnist.py $common --trainer JTT > JTT_log.txt

# Ablation methods for the final-project innovation.
python exp-mnist.py $common --trainer SoftJTT --softjtt_alpha 1.0 > SoftJTT_log.txt
python exp-mnist.py $common --trainer NeutralizedSoftJTT --softjtt_alpha 1.0 --snc_neutralize_power 1.0 > NeutralizedSoftJTT_log.txt
python exp-mnist.py $common --trainer SNCJTT --softjtt_alpha 1.0 --snc_neutralize_power 1.0 --snc_consistency_lambda 0.1 > SNCJTT_log.txt
python exp-mnist.py $common --trainer DS-SNCJTT --softjtt_alpha 1.0 --snc_neutralize_power 1.0 --snc_consistency_lambda 0.1 --ds_num_bins 3 > DSSNCJTT_log.txt
