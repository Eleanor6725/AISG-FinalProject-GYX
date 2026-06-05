#!/bin/bash
# Quick DS-SNCJTT sanity checks.  The default DS implementation now uses
# conditional difficulty bins, which avoids missing attr sides in each stratum.
set -e
seed=516
common="--seed $seed --no_cuda --repeat 1 --iteration 400 --jtt_stage1_steps 100 --softjtt_alpha 1.0 --snc_neutralize_power 1.0 --snc_consistency_lambda 0.1"

python exp-mnist.py $common --trainer SNCJTT > quick_SNCJTT_log.txt
python exp-mnist.py $common --trainer DS-SNCJTT --ds_num_bins 3 --ds_bin_mode conditional > quick_DSSNCJTT_cond_b3_log.txt
python exp-mnist.py $common --trainer DS-SNCJTT --ds_num_bins 4 --ds_bin_mode conditional > quick_DSSNCJTT_cond_b4_log.txt
# Keep this only as a diagnostic to reproduce the old failure mode.
python exp-mnist.py $common --trainer DS-SNCJTT --ds_num_bins 4 --ds_bin_mode global > quick_DSSNCJTT_global_b4_log.txt
