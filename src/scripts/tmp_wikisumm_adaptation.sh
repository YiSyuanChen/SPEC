 #!/bin/bash
export CUDA_VISIBLE_DEVICES=0

##### Manual Settings #####
INSERT_CONDITIONAL_ADAPTER=False
ADAPTER_POSITIONS="full_dec"
ADAPTER_TYPE="default"

TRAIN_PARAMS_CONFIG="full_model"
SURFIX="_eval_best"

TRAIN_NUM=10
START_GROUP=1
END_GROUP=1

##### Corresponding Settings #####
OUTPUT_FOLDER="../results/wikisumm/adaptation/${TRAIN_PARAMS_CONFIG}"

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

INTER_DATASET="cnn_dailymail_wiki_own"
INTER_MODEL_FOLDER="../results/wikisumm/train/full_model/cnn_dailymail_wiki_own_eval_amazon_reviews_multi/checkpoint-100"
TRAIN_EVAL_DATASET="amazon_reviews_multi"

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${INTER_MODEL_FOLDER} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/logs/ \
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

INTER_DATASET="cnn_dailymail_wiki_own"
INTER_MODEL_FOLDER="../results/wikisumm/train/full_model/cnn_dailymail_wiki_own_eval_scitldr/checkpoint-400"
TRAIN_EVAL_DATASET="scitldr"

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${INTER_MODEL_FOLDER} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/logs/ \
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

INTER_DATASET="cnn_dailymail_wiki_own"
INTER_MODEL_FOLDER="../results/wikisumm/train/full_model/cnn_dailymail_wiki_own_eval_samsum/checkpoint-1000"
TRAIN_EVAL_DATASET="samsum"

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${INTER_MODEL_FOLDER} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/logs/ \
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

INTER_DATASET="xsum_wiki_own"
INTER_MODEL_FOLDER="../results/wikisumm/train/full_model/xsum_wiki_own_eval_rottentomatoes_own/checkpoint-1300"
TRAIN_EVAL_DATASET="rottentomatoes_own"

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${INTER_MODEL_FOLDER} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/logs/ \
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


##### Manual Settings #####
INSERT_CONDITIONAL_ADAPTER=False
ADAPTER_POSITIONS="full_dec"
ADAPTER_TYPE="default"

TRAIN_PARAMS_CONFIG="full_model"
SURFIX="_eval_best"

TRAIN_NUM=100
START_GROUP=1
END_GROUP=1

##### Corresponding Settings #####
OUTPUT_FOLDER="../results/wikisumm/adaptation/${TRAIN_PARAMS_CONFIG}"

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

INTER_DATASET="cnn_dailymail_wiki_own"
INTER_MODEL_FOLDER="../results/wikisumm/train/full_model/cnn_dailymail_wiki_own_eval_amazon_reviews_multi/checkpoint-100"
TRAIN_EVAL_DATASET="amazon_reviews_multi"

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${INTER_MODEL_FOLDER} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/logs/ \
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

INTER_DATASET="cnn_dailymail_wiki_own"
INTER_MODEL_FOLDER="../results/wikisumm/train/full_model/cnn_dailymail_wiki_own_eval_scitldr/checkpoint-400"
TRAIN_EVAL_DATASET="scitldr"

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${INTER_MODEL_FOLDER} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/logs/ \
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

INTER_DATASET="cnn_dailymail_wiki_own"
INTER_MODEL_FOLDER="../results/wikisumm/train/full_model/cnn_dailymail_wiki_own_eval_samsum/checkpoint-1000"
TRAIN_EVAL_DATASET="samsum"

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${INTER_MODEL_FOLDER} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/logs/ \
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

INTER_DATASET="xsum_wiki_own"
INTER_MODEL_FOLDER="../results/wikisumm/train/full_model/xsum_wiki_own_eval_rottentomatoes_own/checkpoint-1300"
TRAIN_EVAL_DATASET="rottentomatoes_own"

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${INTER_MODEL_FOLDER} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/logs/ \
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
