#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

##### Manual Settings #####
MODEL_NAME="ainize/bart-base-cnn"
FOLDER_NAME="bart-base-cnn"

TRAIN_PARAMS_CONFIG="full_model"

START_GROUP=1
END_GROUP=5

##### Corresponding Settings #####
OUTPUT_FOLDER="../results/baselines/${FOLDER_NAME}/${TRAIN_PARAMS_CONFIG}"

##### Testing for 10 examples #####
TRAIN_NUM=10

TRAIN_EVAL_DATASET="xsum"
CKPT=(24 24 24 24 24)

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path ${MODEL_NAME} \
		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/checkpoint-${CKPT[$(($g-1))]} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/checkpoint-${CKPT[$(($g-1))]} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/checkpoint-${CKPT[$(($g-1))]}/logs \
		--report_to tensorboard \
		--overwrite_output_dir \
		--per_device_eval_batch_size 16 \
		--do_predict \
		--predict_with_generate \
		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
		#####
done

##### Testing for 100 examples #####
TRAIN_NUM=100
