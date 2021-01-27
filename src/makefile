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

cli_job_im2latex:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset toy_50k \
		--samples 5 \
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
		--epochs 10 \
		--conv-encoder-epochs 2 \
		--batch-size 16 \
		--optimizer adam \
		--lr-schedule vaswani' \
	--workspace .

cli_job_im2latex:
	gradient experiments run singlenode \
	--name latexrec \
	--projectId prqyjc7d5 \
	--container danipozo/latexrec \
	--machineType 'P6000' \
	--command 'python train.py \
		--dataset toy_50k \
		--samples 0 \
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
		--epochs 10 \
		--conv-encoder-epochs 2 \
		--batch-size 16 \
		--optimizer adam \
		--lr-schedule vaswani \
		--log-folder-name toy_basic' \
	--workspace .

cli_job_im2latex_no_ambiguities:
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
		--epochs 10 \
		--conv-encoder-epochs 2 \
		--batch-size 16 \
		--optimizer adam \
		--lr-schedule vaswani \
		--log-folder-name toy_without_ambiguities' \
	--workspace .
