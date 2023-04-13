#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

##### Manual Settings #####
INTER_DATASET="amazon_reviews_multi_unsup_own"
EVAL_DATASET="amazon_reviews_multi_cond_own"

INSERT_CONDITIONAL_ADAPTER=True
ADAPTER_POSITIONS="full_dec"
ADAPTER_TYPE="default"

TRAIN_PARAMS_CONFIG="full_model_with_full_dec_ada"
SURFIX="_eval_best"

EPOCHS=9
EVAL_SAVE_STEPS=100

##### Device-Dependent Settings #####
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=16
GRAD_ACCUM=8
PREPRO_WORKERS=8

##### Corresponding Settings #####
OUTPUT_FOLDER="../results/spec_ipl/train/${TRAIN_PARAMS_CONFIG}"

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

##### Training #####
python main.py \
	--dataset_name ${INTER_DATASET} \
	--eval_dataset_name ${EVAL_DATASET} \
	--model_name_or_path facebook/bart-base \
	--output_dir ${OUTPUT_FOLDER}/${INTER_DATASET}_eval_${EVAL_DATASET}${SURFIX} \
	--logging_dir ${OUTPUT_FOLDER}/${INTER_DATASET}_eval_${EVAL_DATASET}${SURFIX}/logs/ \
	--report_to tensorboard \
	--overwrite_output_dir \
	--evaluation_strategy steps \
	--save_strategy steps \
	--num_train_epochs ${EPOCHS} \
	--eval_steps ${EVAL_SAVE_STEPS} \
	--save_steps ${EVAL_SAVE_STEPS} \
	--save_total_limit 1 \
	--logging_steps 1 \
	--per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
	--per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
	--lr_scheduler_type constant \
	--learning_rate 3e-5 \
	--do_train \
	--do_eval \
	--gradient_accumulation_steps ${GRAD_ACCUM} \
	--max_val_samples 1000 \
	--label_smoothing_factor 0.1 \
	--weight_decay 0.01 \
	--max_grad_norm 0.1 \
	--warmup_ratio 0 \
	--preprocessing_num_workers ${PREPRO_WORKERS} \
	--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
	--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
	--adapter_positions ${ADAPTER_POSITIONS} \
	--adapter_type ${ADAPTER_TYPE} \
	--predict_with_generate \
	--save_model_accord_to_rouge \
