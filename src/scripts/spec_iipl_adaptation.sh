 #!/bin/bash
export CUDA_VISIBLE_DEVICES=0

##### Manual Settings #####
META_DATASET=("aeslc_cond_own" "reddit_tifu_cond_own" "amazon_reviews_multi_unsup_own")
META_MODEL_FOLDER="../results/spec_iipl/train/full_model_with_full_dec_sw_ada/aeslc_cond_own_reddit_tifu_cond_own_amazon_reviews_multi_unsup_own_eval_amazon_reviews_multi_cond_own/checkpoint-9400"
TRAIN_EVAL_DATASET="amazon_reviews_multi_cond_own"

INSERT_CONDITIONAL_ADAPTER=True
ADAPTER_POSITIONS="full_dec"
ADAPTER_TYPE="sw"

TRAIN_PARAMS_CONFIG="full_model_with_full_dec_${ADAPTER_TYPE}_ada"
SURFIX="_eval_best"

TRAIN_NUM=10
START_GROUP=1
END_GROUP=2

##### Corresponding Settings #####
OUTPUT_FOLDER="../results/spec_iipl/adaptation/${TRAIN_PARAMS_CONFIG}"

join_arr() {
	local IFS="$1"
	shift
	echo "$*"
}

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

##### Sanity Check #####
if [ $INSERT_CONDITIONAL_ADAPTER == True ]
then
	if ! [[ $TRAIN_PARAMS_CONFIG == *"ada"* ]]
	then
		echo "Using conditional module but the parameter config is not 'ada'."
		exit 1
	fi
	
	if ! [[ $TRAIN_EVAL_DATASET == *"cond"* ]]
	then
		echo "Using conditional module but the dataset is not 'cond'."
		exit 1
	fi	
else
	if [[ $TRAIN_PARAMS_CONFIG == *"ada"* ]]
	then
		echo "Not using conditional module but the parameter config is 'ada'."
		exit 1
	fi

	if [[ $TRAIN_EVAL_DATASET == *"cond"* ]]
	then
		echo "Not using conditional module but the dataset is 'cond'."
		exit 1
	fi	
fi

##### Training #####

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${META_MODEL_FOLDER} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/$(join_arr _ "${META_DATASET[@]}")${SURFIX} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/$(join_arr _ "${META_DATASET[@]}")${SURFIX}/logs/ \
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
		--insert_conditional_adapters ${INSERT_CONDITIONAL_ADAPTER} \
		--adapter_positions ${ADAPTER_POSITIONS} \
		--adapter_type ${ADAPTER_TYPE} \
		--predict_with_generate \
		--save_model_accord_to_rouge \
		#####
done
