export CUDA_VISIBLE_DEVICES=0
DATASET=amazon_reviews_multi

########## Unsupervised Dataset ##########
#python condition_dataset.py \
#	--dataset $DATASET \
#	--train_file ../../datasets/unsupervised/$DATASET/train.csv \
#	--validation_file ../../datasets/unsupervised/$DATASET/validation.csv \
#	--test_file ../../datasets/unsupervised/$DATASET/test.csv \
#	--output_dir ../../datasets/unsupervised/try/$DATASET \
#	--summ_sent_num 1 \
#	--max_train_samples 100000 \
#	#--max_val_samples 1000 \
#	#--max_test_samples 1000 \

########## Supervised Dataset ##########
#python condition_dataset.py \
#	--dataset $DATASET \
#	--train_file ../../datasets/supervised/${DATASET}/train.csv \
#	--validation_file ../../datasets/supervised/${DATASET}/validation.csv \
#	--test_file ../../datasets/supervised/${DATASET}/test.csv \
#	--output_dir ../../datasets/supervised/try/${DATASET} \
#	--use_ground_truth \
#	#--max_train_samples 1000 \
#	#--max_val_samples 1000 \
#	#--max_test_samples 1000 \

########## Condition Analysis ##########
#DATASET=amazon_reviews_multi
#python analysis.py \
#	--dataset_file_1 ../../datasets/supervised/$DATASET/train.csv \
#	--name_1 $DATASET \
#	--output_dir ../../analysis/try/$DATASET \
#	--perplexity 30 \
#	--n_iter 5000 \
#	--max_samples 100 \
#	--num_cluster 8 \
