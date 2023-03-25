 #!/bin/bash
export CUDA_VISIBLE_DEVICES=0

##### Manual Settings #####
INSERT_CONDITIONAL_ADAPTER=False
ADAPTER_POSITIONS="full_dec"
ADAPTER_TYPE="default"

TRAIN_PARAMS_CONFIG="full_model"
SURFIX="_eval_best"

START_GROUP=1
END_GROUP=1

##### Corresponding Settings #####
OUTPUT_FOLDER="../results/wikisumm/adaptation/${TRAIN_PARAMS_CONFIG}"

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
TRAIN_NUM=10

INTER_DATASET="xsum_wiki_own"
TRAIN_EVAL_DATASET="xsum"
CKPT=(16)

for g in $(seq $START_GROUP $END_GROUP)
do
	python main.py \
		--dataset_name ${TRAIN_EVAL_DATASET} \
		--model_name_or_path facebook/bart-base \
		--load_trained_model_from ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
		--output_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]} \
		--logging_dir ${OUTPUT_FOLDER}/${TRAIN_EVAL_DATASET}_${TRAIN_NUM}_examples/group_${g}/${INTER_DATASET}${SURFIX}/checkpoint-${CKPT[$(($g-1))]}/logs \
		--report_to tensorboard \
		--overwrite_output_dir \
		--per_device_eval_batch_size 16 \
		--do_predict \
		--predict_with_generate \
		--trainable_params_config ${TRAIN_PARAMS_CONFIG} \
		####
done

###### Testing 100 Examples #####
TRAIN_NUM=100
