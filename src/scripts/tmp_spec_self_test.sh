 #!/bin/bash
export CUDA_VISIBLE_DEVICES=0

##### Manual Settings #####
INSERT_CONDITIONAL_ADAPTER=True
ADAPTER_POSITIONS="full_dec"
ADAPTER_TYPE="default"

TRAIN_PARAMS_CONFIG="full_model_with_full_dec_ada"
SURFIX="_eval_best"

START_GROUP=1
END_GROUP=1

##### Corresponding Settings #####
OUTPUT_FOLDER="../results/spec_self/adaptation/${TRAIN_PARAMS_CONFIG}"

##### Sanity Check #####
if [ $INSERT_CONDITIONAL_ADAPTER == True ]
then
	if ! [[ $TRAIN_PARAMS_CONFIG == *"ada"* ]]
	then
		echo "Using conditional module but the parameter config is not 'ada'."
		exit 1
	fi
	
else
	if [[ $TRAIN_PARAMS_CONFIG == *"ada"* ]]
	then
		echo "Not using conditional module but the parameter config is 'ada'."
		exit 1
	fi
fi

##### Testing for 10 examples #####
TRAIN_NUM=100
CLUSTER_DATA_NUM=100
CLUSTER_NUM=8
DECODE_METHOD="test_cluster_${CLUSTER_DATA_NUM}_examples_${CLUSTER_NUM}_clusters"
BIAS_SURFIX=""

INTER_DATASET="xsum_unsup_own"
TRAIN_EVAL_DATASET="xsum_cond_own"
TURE_NAME="xsum"
CKPT=(45)

#INTER_DATASET="reddit_tifu_unsup_own"
#TRAIN_EVAL_DATASET="reddit_tifu_cond_own"
#TURE_NAME="reddit_tifu"
#CKPT=(44)

#INTER_DATASET="amazon_reviews_multi_unsup_own"
#TRAIN_EVAL_DATASET="amazon_reviews_multi_cond_own"
#TURE_NAME="amazon_reviews_multi"
#CKPT=(22)

#INTER_DATASET="scitldr_unsup_own"
#TRAIN_EVAL_DATASET="scitldr_cond_own"
#TURE_NAME="scitldr"
#CKPT=(16)

#INTER_DATASET="samsum_unsup_own"
#TRAIN_EVAL_DATASET="samsum_cond_own"
#TURE_NAME="samsum"
#CKPT=(20)

#INTER_DATASET="rottentomatoes_unsup_own"
#TRAIN_EVAL_DATASET="rottentomatoes_cond_own"
#TURE_NAME="rottentomatoes"
#CKPT=(20)

OUTPUT_FOLDER_TMP="../results/reply/zero_shot/spec_self/adaptation/${TRAIN_PARAMS_CONFIG}"
DECODE_METHOD="test_cluster_${CLUSTER_DATA_NUM}_examples_${CLUSTER_NUM}_clusters/${TURE_NAME}"

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name reddit_tifu_cond_own \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
		--output_dir ${OUTPUT_FOLDER_TMP}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
		--logging_dir ${OUTPUT_FOLDER_TMP}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
		--report_to tensorboard \
		--overwrite_output_dir \
		--per_device_eval_batch_size 16 \
		--do_predict \
		--predict_with_generate \
		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
		--adapter_positions ${ADAPTER_POSITIONS} \
		--adapter_type ${ADAPTER_TYPE} \
		--cluster_conditions_path ../../ConditionDataset/analysis/reddit_tifu/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
        --max_test_samples 1000 \
		#--dataset_name ${TRAIN_EVAL_DATASET} \
		#--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
		####
done

#INTER_DATASET="reddit_tifu_unsup_own"
#TRAIN_EVAL_DATASET="reddit_tifu_cond_own"
#TURE_NAME="reddit_tifu"
#CKPT=(44)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
#
#INTER_DATASET="amazon_reviews_multi_unsup_own"
#TRAIN_EVAL_DATASET="amazon_reviews_multi_cond_own"
#TURE_NAME="amazon_reviews_multi"
#CKPT=(22)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
#
#INTER_DATASET="scitldr_unsup_own"
#TRAIN_EVAL_DATASET="scitldr_cond_own"
#TURE_NAME="scitldr"
#CKPT=(16)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
#
#INTER_DATASET="samsum_unsup_own"
#TRAIN_EVAL_DATASET="samsum_cond_own"
#TURE_NAME="samsum"
#CKPT=(20)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
##
#INTER_DATASET="rottentomatoes_unsup_own"
#TRAIN_EVAL_DATASET="rottentomatoes_cond_own"
#TURE_NAME="rottentomatoes"
#CKPT=(20)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
#
##INTER_DATASET="scitldr_unsup_own"
##TRAIN_EVAL_DATASET="scitldr_except_first_targets_cond_own"
##TURE_NAME="scitldr_except_first_targets"
##CKPT=(16)
##
##for g in $(seq $START_GROUP $END_GROUP)
##do
##	python main.py \
##		--dataset_name ${TRAIN_EVAL_DATASET} \
##		--model_name_or_path facebook/bart-base \
##		--load_trained_model_from ${OUTPUT_FOLDER}/scitldr_cond_own_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
##		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
##		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
##		--report_to tensorboard \
##		--overwrite_output_dir \
##		--per_device_eval_batch_size 16 \
##		--do_predict \
##		--predict_with_generate \
##		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
##		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
##		--adapter_positions ${ADAPTER_POSITIONS} \
##		--adapter_type ${ADAPTER_TYPE} \
##		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
##		####
##done
#
####### Testing 100 Examples #####
#TRAIN_NUM=100
#CLUSTER_DATA_NUM=100
#CLUSTER_NUM=16
#DECODE_METHOD="test_cluster_${CLUSTER_DATA_NUM}_examples_${CLUSTER_NUM}_clusters"
#BIAS_SURFIX=""
#
#INTER_DATASET="xsum_unsup_own"
#TRAIN_EVAL_DATASET="xsum_cond_own"
#TURE_NAME="xsum"
#CKPT=(45)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
#
#INTER_DATASET="reddit_tifu_unsup_own"
#TRAIN_EVAL_DATASET="reddit_tifu_cond_own"
#TURE_NAME="reddit_tifu"
#CKPT=(40)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
#
#INTER_DATASET="amazon_reviews_multi_unsup_own"
#TRAIN_EVAL_DATASET="amazon_reviews_multi_cond_own"
#TURE_NAME="amazon_reviews_multi"
#CKPT=(90)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
#
#INTER_DATASET="scitldr_unsup_own"
#TRAIN_EVAL_DATASET="scitldr_cond_own"
#TURE_NAME="scitldr"
#CKPT=(40)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
#
#INTER_DATASET="samsum_unsup_own"
#TRAIN_EVAL_DATASET="samsum_cond_own"
#TURE_NAME="samsum"
#CKPT=(100)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
#
#INTER_DATASET="rottentomatoes_unsup_own"
#TRAIN_EVAL_DATASET="rottentomatoes_cond_own"
#TURE_NAME="rottentomatoes"
#CKPT=(30)
#
#for g in $(seq $START_GROUP $END_GROUP)
#do
#	python main.py \
#		--dataset_name ${TRAIN_EVAL_DATASET} \
#		--model_name_or_path facebook/bart-base \
#		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
#		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
#		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
#		--report_to tensorboard \
#		--overwrite_output_dir \
#		--per_device_eval_batch_size 16 \
#		--do_predict \
#		--predict_with_generate \
#		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
#		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
#		--adapter_positions ${ADAPTER_POSITIONS} \
#		--adapter_type ${ADAPTER_TYPE} \
#		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
#		####
#done
#
##TRAIN_NUM=100
##CLUSTER_DATA_NUM=100
##CLUSTER_NUM=8
##TYPE_SURFIX="_cond_7"
##DECODE_METHOD="test_cluster_${CLUSTER_DATA_NUM}_examples_${CLUSTER_NUM}_clusters${TYPE_SURFIX}"
##
##INTER_DATASET="xsum_unsup_own"
##TRAIN_EVAL_DATASET="xsum_example_own"
##TURE_NAME="xsum"
##CKPT=(45)
##
##for g in $(seq $START_GROUP $END_GROUP)
##do
##	python main.py \
##		--dataset_name ${TRAIN_EVAL_DATASET} \
##		--model_name_or_path facebook/bart-base \
##		--load_trained_model_from ${OUTPUT_FOLDER}/xsum_cond_own_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
##		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
##		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
##		--report_to tensorboard \
##		--overwrite_output_dir \
##		--per_device_eval_batch_size 16 \
##		--do_predict \
##		--predict_with_generate \
##		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
##		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
##		--adapter_positions ${ADAPTER_POSITIONS} \
##		--adapter_type ${ADAPTER_TYPE} \
##		--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/for_examples/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${TYPE_SURFIX}.npy \
##		#--cluster_conditions_path ../../ConditionDataset/analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_${CLUSTER_NUM}_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
##		####
##done
