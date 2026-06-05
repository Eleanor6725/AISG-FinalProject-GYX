#!/bin/bash
# Quick smoke / tuning script. Uses repeat=1 and hides dataset diagnostics after the first run.
seed=516
common="--seed $seed --no_cuda --repeat 1 --iteration 400 --jtt_stage1_steps 100 --jtt_no_eval_stage1 --no_check_data"

python exp-mnist.py $common --trainer JTT --jtt_up_weight 20 > quick_JTT_log.txt
python exp-mnist.py $common --trainer SoftJTT --softjtt_alpha 0.5 > quick_SoftJTT_a0.5_log.txt
python exp-mnist.py $common --trainer SoftJTT --softjtt_alpha 1.0 > quick_SoftJTT_a1_log.txt
python exp-mnist.py $common --trainer NeutralizedSoftJTT --softjtt_alpha 1.0 --snc_neutralize_power 1.0 > quick_NeutralizedSoftJTT_log.txt
python exp-mnist.py $common --trainer SNCJTT --softjtt_alpha 1.0 --snc_neutralize_power 1.0 --snc_consistency_lambda 0.1 > quick_SNCJTT_log.txt
