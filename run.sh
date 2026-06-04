#!/bin/bash
# chmod +x run.sh
seed=516
python exp-mnist.py --seed $seed --no_cuda --trainer ERM > ERM_log.txt
python exp-mnist.py --seed $seed --no_cuda --reg_name IRM --trainer ERM > IRM_log.txt
python exp-mnist.py --seed $seed --no_cuda --trainer groupDRO > groupDRO_log.txt
python exp-mnist.py --seed $seed --no_cuda --trainer JTT --jtt_stage1_steps 100 --jtt_up_weight 20 > JTT_log.txt
