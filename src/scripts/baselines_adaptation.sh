#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

##### Manual Settings #####
#TRAIN_EVAL_DATASET_LIST=("xsum" "reddit_tifu" "scitldr" "samsum" "amazon_reviews_multi" "rottentomatoes_own")
TRAIN_EVAL_DATASET_LIST=("reddit_tifu" "samsum")

MODEL_NAME="ainize/bart-base-cnn"
FOLDER_NAME="bart-base-cnn"

TRAIN_PARAMS_CONFIG="full_model"

TRAIN_NUM=10
START_GROUP=1
END_GROUP=1

##### Corresponding Settings #####
OUTPUT_FOLDER="../results/baselines/${FOLDER_NAME}/${TRAIN_PARAMS_CONFIG}"

if [ $TRAIN_NUM == 10 ]
then
	MAX_STEPS=100
	EVAL_SAVE_STEPS=2
	TRAIN_BATCH_SIZE=2
	GRAD_ACCUM=4
	WARMUP_STEPS=25

elif [ $TRAIN_NUM == 100 ]
then
	MAX_STEPS=200
	EVAL_SAVE_STEPS=5
	TRAIN_BATCH_SIZE=2
	GRAD_ACCUM=16
	WARMUP_STEPS=20
fi

##### Training #####
for TRAIN_EVAL_DATASET in ${TRAIN_EVAL_DATASET_LIST[@]}
do
	for g in $(seq $START_GROUP $END_GROUP)
	do
		python main.py \
			--dataset_name ${TRAIN_EVAL_DATASET} \
			--model_name_or_path ${MODEL_NAME} \
			--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g} \
			--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/logs/ \
			--report_to tensorboard \
			--overwrite_output_dir \
			--evaluation_strategy steps \
			--save_strategy steps \
			--max_steps ${MAX_STEPS} \
			--eval_steps ${EVAL_SAVE_STEPS} \
			--save_steps ${EVAL_SAVE_STEPS} \
			--save_total_limit 1 \
			--logging_steps 1 \
			--per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
			--per_device_eval_batch_size 16 \
			--lr_scheduler_type polynomial \
			--learning_rate 3e-5 \
			--do_train \
			--do_eval \
			--gradient_accumulation_steps ${GRAD_ACCUM} \
			--max_train_samples ${TRAIN_NUM} \
			--max_val_samples 1000 \
			--label_smoothing_factor 0.1 \
			--weight_decay 0.01 \
			--max_grad_norm 0.1 \
			--warmup_steps ${WARMUP_STEPS} \
			--select_start_indice $(( ($g-1)*$TRAIN_NUM )) \
			--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
			--predict_with_generate \
			--save_model_accord_to_rouge \
			#####
	done
done

