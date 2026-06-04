#!/bin/bash
# chmod +x run.sh
# seed=516
# python exp-mnist.py  --seed $seed --no_cuda --trainer ERM > ERM_log.txt
# python exp-mnist.py  --seed $seed --no_cuda --reg_name IRM --trainer ERM > IRM_log.txt
# python exp-mnist.py  --seed $seed --no_cuda --trainer groupDRO > groupDRO_log.txt


#!/bin/bash
# chmod +x run.sh
seed=516
COMMON="--seed $seed --no_cuda --metric_list Accuracy AUC F1_macro GroupAccuracy WorstGroupAccuracy GroupGap"

python exp-mnist.py $COMMON --trainer ERM > ERM_log.txt
python exp-mnist.py $COMMON --reg_name IRM --trainer ERM > IRM_log.txt
python exp-mnist.py $COMMON --trainer groupDRO --groupdro_group_by group > groupDRO_log.txt
