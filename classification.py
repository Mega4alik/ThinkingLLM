# classification reasoning through looping. m1) k=4, L=4
# venv: US1-asr3.12

import json
import os
import numpy as np
import random
import time
import datetime
import copy
from datasets import load_dataset, Dataset
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

def messages_to_prompt(messages):
	return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

def preprocess(batch):
	prompts, labels = [], []
	for i in range(len(batch["question"])): #question, answer, knowledge, hallucination
		messages = [
			{"role": "system", "content": batch["knowledge"][i]},
			{"role": "user", "content": batch["question"][i]},
			{"role": "assistant", "content": batch["answer"][i]+"\nIs this answer supported by the context above?"}
		]
		prompt = messages_to_prompt(messages)
		#print(prompt,  tokenizer.decode(prompt), len(prompt)); exit()
		prompts.append(prompt)
		label = 1 if batch["hallucination"][i]=="yes" else 0
		labels.append(label)
	return {"prompts":prompts, "labels":labels}


class myDataCollator:
	def __call__(self, features):
		input_ids, labels = [], []
		for x in features:
			prompt_tokens, label = x["prompts"], x["labels"]
			input_ids.append(torch.tensor(prompt_tokens))
			labels.append(label)

		input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
		labels = torch.tensor(labels)
		#labels = pad_sequence(labels, batch_first=True, padding_value=-100)
		#attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
		#print("data_collator:", input_ids.shape, labels.shape)
		return {"input_ids": input_ids, "labels": labels}


def get_averaged_layers(layers, k):
	# Number of groups: 4, each with 6 layers
	new_layers = nn.ModuleList()
	L = len(layers)//k
	for i in range(0, len(layers), L):
		# Get 6 layers to average
		group = layers[i:i+L]
		# Deepcopy the first one as the base to hold averaged weights
		avg_layer = copy.deepcopy(group[0])
		with torch.no_grad():
			for name, param in avg_layer.named_parameters():
				# Sum up corresponding parameters
				stacked_params = torch.stack([layer.state_dict()[name] for layer in group])
				avg_param = stacked_params.mean(dim=0)
				param.copy_(avg_param)

		new_layers.append(avg_layer)

	return new_layers
	

#==============================================================================================
if __name__ == "__main__":
	model_id = "Qwen/Qwen2-0.5B-Instruct" #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  #"meta-llama/Llama-3.2-1B-Instruct"  #"meta-llama/Llama-3.2-1B"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	tokenizer.pad_token = tokenizer.eos_token #'!' #'<|finetune_right_pad_id|>' 
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.truncation_side = 'left'
	print("tokenizer:", tokenizer.pad_token_id, tokenizer.truncation_side)	

	if 1==2: #training start
		model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
		model.config.pad_token_id = tokenizer.pad_token_id
		# looping L=24/4(k)
		model.config.num_hidden_layers=4
		#model.model.layers = model.model.layers[:4]
		model.model.layers = get_averaged_layers(model.model.layers, 4)
		#endOf looping
	else:
		model = AutoModelForSequenceClassification.from_pretrained("./model_temp/checkpoint-1000")
		#print(model.config);exit()

	# Dataset
	dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
	dataset = dataset.map(preprocess, batched=True)
	dataset = dataset.filter(lambda x: len(x["prompts"]) <= 200)
	dataset = dataset.train_test_split(test_size=0.01, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print("Dataset train, test sizes:",  len(train_dataset), len(test_dataset))	
	
	# Start training
	data_collator = myDataCollator()
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=50,
		per_device_train_batch_size=8,
		gradient_accumulation_steps=1,
		#gradient_checkpointing=True, - slows down the training
		learning_rate=1e-6,
		logging_steps=20,
		save_steps=500,
		save_total_limit=3,
		load_best_model_at_end=True,
		eval_strategy="steps",
		eval_steps=500,
		per_device_eval_batch_size=1,
		remove_unused_columns=False,
		max_grad_norm=100.0,  # 🚨 This enables gradient clipping
		#logging_dir="./logs/",
		#report_to="tensorboard",
		#weight_decay=0.01,
	)

	trainer = Trainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset		
	)
		
	trainer.train("./model_temp/checkpoint-1000")

	#print( trainer.evaluate() )

