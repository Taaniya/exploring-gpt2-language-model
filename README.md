# exploring-gpt2-language-model

Finetune Transformer decoder only models like GPT2 with causal language model objective

```
python finetune_gpt2_clm_pytorch.py \
	--model-name gpt2 \
	--dataset-path train.txt \
	--batch-size 8 \
	--epoch 4 \
	--model-version-no 1 \
	--model-dir ./
```  
