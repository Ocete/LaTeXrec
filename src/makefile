all: test_script

test_script:
	python train.py \
	--dataset toy_50k \
	--samples 1000 \
	--remove-ambiguities yes \
	--num-layers 1 \
	--depth 20 \
	--feedforward-units 32 \
	--num-heads 1 \
	--dropout-rate 0.1 \
	--performer-attention-encoder no \
	--positional-encoding standard \
	--conv-encoder vanilla \
	--conv-filters 32 \
	--pretrain-conv-encoder no \
	--maximum-target-length 200 \
	--epochs 2 \
	--conv-encoder-epochs 2 \
	--batch-size 16 \
	--optimizer adam \
	--lr-schedule vaswani \
	--log-folder-name test1

cli_job_im2latex_wed_1:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset im2latex \
		--remove-ambiguities no \
		--num-layers 1 \
		--depth 20 \
		--feedforward-units 32 \
		--num-heads 1 \
		--dropout-rate 0.1 \
		--performer-attention-encoder no \
		--positional-encoding standard \
		--conv-encoder vanilla \
		--conv-filters 32 \
		--pretrain-conv-encoder no \
		--maximum-target-length 200 \
		--epochs 2 \
		--conv-encoder-epochs 2 \
		--batch-size 32 \
		--optimizer adam \
		--lr-schedule vaswani \
		--log-folder-name wednesday_1' \
	--workspace .

cli_job_im2latex_wed_2:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset im2latex \
		--remove-ambiguities yes \
		--num-layers 1 \
		--depth 20 \
		--feedforward-units 32 \
		--num-heads 1 \
		--dropout-rate 0.1 \
		--performer-attention-encoder no \
		--positional-encoding standard \
		--conv-encoder vanilla \
		--conv-filters 32 \
		--pretrain-conv-encoder no \
		--maximum-target-length 200 \
		--epochs 2 \
		--conv-encoder-epochs 2 \
		--batch-size 32 \
		--optimizer adam \
		--lr-schedule vaswani \
		--log-folder-name wednesday_2' \
	--workspace .

cli_job_im2latex_wed_3a:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset im2latex \
		--remove-ambiguities yes \
		--num-layers 1 \
		--depth 20 \
		--feedforward-units 32 \
		--num-heads 1 \
		--dropout-rate 0.1 \
		--performer-attention-encoder yes \
		--positional-encoding standard \
		--conv-encoder vanilla \
		--conv-filters 32 \
		--pretrain-conv-encoder no \
		--maximum-target-length 200 \
		--epochs 2 \
		--conv-encoder-epochs 2 \
		--batch-size 32 \
		--optimizer adam \
		--lr-schedule vaswani \
		--log-folder-name wednesday_3a' \
	--workspace .

cli_job_im2latex_wed_3b:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset im2latex \
		--remove-ambiguities yes \
		--num-layers 1 \
		--depth 20 \
		--feedforward-units 32 \
		--num-heads 1 \
		--dropout-rate 0.1 \
		--performer-attention-encoder no \
		--positional-encoding standard \
		--conv-encoder resnet \
		--conv-filters 32 \
		--pretrain-conv-encoder no \
		--maximum-target-length 200 \
		--epochs 2 \
		--conv-encoder-epochs 2 \
		--batch-size 32 \
		--optimizer adam \
		--lr-schedule vaswani \
		--log-folder-name wednesday_3b' \
	--workspace .

cli_job_im2latex_wed_3c:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset im2latex \
		--remove-ambiguities yes \
		--num-layers 1 \
		--depth 20 \
		--feedforward-units 32 \
		--num-heads 1 \
		--dropout-rate 0.1 \
		--performer-attention-encoder no \
		--positional-encoding 2d \
		--conv-encoder vanilla \
		--conv-filters 32 \
		--pretrain-conv-encoder no \
		--maximum-target-length 200 \
		--epochs 2 \
		--conv-encoder-epochs 2 \
		--batch-size 32 \
		--optimizer adam \
		--lr-schedule vaswani \
		--log-folder-name wednesday_3c' \
	--workspace .

cli_job_im2latex_wed_big_boy:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset im2latex \
		--remove-ambiguities yes \
		--num-layers 1 \
		--depth 20 \
		--feedforward-units 32 \
		--num-heads 1 \
		--dropout-rate 0.1 \
		--performer-attention-encoder yes \
		--positional-encoding 2d \
		--conv-encoder resnet \
		--conv-filters 32 \
		--pretrain-conv-encoder no \
		--maximum-target-length 200 \
		--epochs 5 \
		--conv-encoder-epochs 2 \
		--batch-size 32 \
		--optimizer adam \
		--lr-schedule vaswani \
		--log-folder-name wednesday_big_boy' \
	--workspace .

cli_job_im2latex_wed_bigger_boy:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset im2latex \
		--remove-ambiguities yes \
		--num-layers 4 \
		--depth 32 \
		--feedforward-units 32 \
		--num-heads 4 \
		--dropout-rate 0.1 \
		--performer-attention-encoder yes \
		--positional-encoding 2d \
		--conv-encoder resnet \
		--conv-filters 32 \
		--pretrain-conv-encoder no \
		--maximum-target-length 200 \
		--epochs 3 \
		--conv-encoder-epochs 2 \
		--batch-size 32 \
		--optimizer adam \
		--lr-schedule vaswani \
		--log-folder-name wednesday_bigger_boy' \
	--workspace .

cli_job_test_bleu:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset toy_50k \
		--remove-ambiguities yes \
		--num-layers 1 \
		--depth 20 \
		--feedforward-units 32 \
		--num-heads 1 \
		--dropout-rate 0.1 \
		--performer-attention-encoder no \
		--positional-encoding standard \
		--conv-encoder vanilla \
		--conv-filters 32 \
		--pretrain-conv-encoder no \
		--maximum-target-length 200 \
		--epochs 3 \
		--conv-encoder-epochs 2 \
		--batch-size 32 \
		--optimizer adam \
		--lr-schedule vaswani' \
	--workspace .


cli_job_im2latex_last_boy:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset im2latex \
		--remove-ambiguities yes \
		--num-layers 4 \
		--depth 20 \
		--feedforward-units 32 \
		--num-heads 4 \
		--dropout-rate 0.1 \
		--performer-attention-encoder yes \
		--positional-encoding 2d \
		--conv-encoder resnet \
		--conv-filters 32 \
		--pretrain-conv-encoder no \
		--maximum-target-length 200 \
		--epochs 3 \
		--conv-encoder-epochs 2 \
		--batch-size 32 \
		--optimizer adam \
		--lr-schedule vaswani \
		--log-folder-name last_boy' \
	--workspace .