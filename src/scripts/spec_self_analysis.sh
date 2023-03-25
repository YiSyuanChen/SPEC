 #!/bin/bash
export CUDA_VISIBLE_DEVICES=0

##### Manual Settings #####
DATASET="xsum_cond_own"
TRUE_DATASET_NAME="xsum"
INTER_DATASET="xsum_unsup_own"

TRAIN_PARAMS_CONFIG="full_model_with_full_dec_ada"
SURFIX="_eval_best"

TRAIN_NUM=10
START_GROUP=1
END_GROUP=1

##### Corresponding Settings #####
OUTPUT_FOLDER="../results/spec_self/adaptation/${TRAIN_PARAMS_CONFIG}"

##### Analysis for Condition Recover #####

#CKPT=(26)
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python analysis_cond_recover.py \
#		--dataset_name ${TRUE_DATASET_NAME} \
#		--gen_file_1 ${OUTPUT_FOLDER}/${DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/test_gold/test_generations.txt \
#		--gen_file_2 ${OUTPUT_FOLDER}/${DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/test_cluster_10_examples_8_clusters/test_generations.txt \
#		--test_file ../../ConditionDataset/datasets/supervised/${TRUE_DATASET_NAME}/test.csv  \
#		--output_dir ../analysis \
#		--debug \
#		--max_samples 100 \
#		#####
#done

##### Analysis for Recall Precision #####

#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python analysis_token_loss.py \
#		--token_loss_file_1 ../results/spec_self/adaptation/full_model_with_full_dec_ada/xsum_cond_own_10_examples/group_1/xsum_unsup_own_eval_best/checkpoint-26/test_gold_token_loss_full_examples/test_loss_per_token.npy \
#		--token_loss_file_2 ../results/spec_self/adaptation/full_model_with_full_dec_ada/xsum_cond_own_10_examples/group_1/xsum_unsup_own_eval_best/checkpoint-26/test_cluster_1000_examples_8_clusters_token_loss_full_examples/test_loss_per_token.npy \
#		--output_dir ../analysis \
#		#####
#done

##### Examples #####

#CKPT=(45)
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python analysis_examples.py \
#		--dataset_name ${TRUE_DATASET_NAME} \
#		--gold_file ../../ConditionDataset/datasets/supervised/${TRUE_DATASET_NAME}/test.csv \
#		--gen_file ${OUTPUT_FOLDER}/${DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/test_cluster_${TRAIN_NUM}_examples_8_clusters/test_generations.txt \
#		--output_dir ../analysis \
#		#####
#done

#### Preference Visualization #####
#python analysis_parallel.py
python analysis_pca.py

##### Example Comparison #####
#python analysis_examples_comparison.py \
#	--dataset_name xsum \
#	--test_file ../analysis/examples/xsum/test.csv \
#	--gen_folder ../results/spec_self/adaptation/full_model_with_full_dec_ada/xsum_example_own_100_examples/group_1/xsum_unsup_own_eval_best/checkpoint-45/ \
#	--output_dir ../analysis \
