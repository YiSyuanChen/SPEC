 #!/bin/bash
export CUDA_VISIBLE_DEVICES=0

##### Manual Settings #####
INSERT_CONDITIONAL_ADAPTER=True
ADAPTER_POSITIONS="full_dec"
ADAPTER_TYPE="default"

TRAIN_PARAMS_CONFIG="full_model_with_full_dec_ada"
SURFIX="_eval_best"

TRAIN_NUM=10 # 100
START_GROUP=1
END_GROUP=2

CLUSTER_DATA_NUM=10 # 100
CLUSTER_NUM=8
DECODE_METHOD="test_cluster_${CLUSTER_DATA_NUM}_examples_${CLUSTER_NUM}_clusters"
BIAS_SURFIX=""

INTER_DATASET="amazon_reviews_multi_unsup_own"
TRAIN_EVAL_DATASET="amazon_reviews_multi_cond_own"
TURE_NAME="amazon_reviews_multi"
CKPT=(6 34)

##### Corresponding Settings #####
OUTPUT_FOLDER="../results/spec_ipl/adaptation/${TRAIN_PARAMS_CONFIG}"

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

##### Testing #####
for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/${DECODE_METHOD}/logs \
		--report_to tensorboard \
		--overwrite_output_dir \
		--per_device_eval_batch_size 16 \
		--do_predict \
		--predict_with_generate \
		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
		--adapter_positions ${ADAPTER_POSITIONS} \
		--adapter_type ${ADAPTER_TYPE} \
		--cluster_conditions_path ../analysis/${TURE_NAME}/means_iter_5000_perp_30_cluster_8_examples_${CLUSTER_DATA_NUM}${BIAS_SURFIX}.npy \
		####
done
