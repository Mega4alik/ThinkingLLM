# reasoning through looping
# venv: asr3.12 - US1

import json
import os
import numpy as np
import random
import time
import datetime
import evaluate
from datasets import load_dataset, Dataset
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer
#from utils import file_get_contents, file_put_contents
from modeling import LoopedLM, LightweightThinkingModel


def messages_to_prompt(messages):
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def preprocess(batch):	
	prompts, labels = [], []
	for i in range(len(batch["question"])): #Problem, Solution, Answer | question, answer
		messages = [
			{"role": "system", "content": "Given math problem, generate final answer. Don't add <think> steps."},
			{"role": "user", "content": batch["question"][i]}
		]
		prompt = messages_to_prompt(messages) #batch["Problem"][i]		
		prompts.append(prompt)
		label = str(batch["answer"][i])
		labels.append(label)
	return {"prompts":prompts, "labels":labels}


class myDataCollator:
	def __call__(self, features):
		prompts = [x["prompts"] for x in features]
		labels = [x["labels"] for x in features]
		labels = tokenizer(labels, padding=True,  truncation=False, return_tensors='pt').input_ids		
		batch = tokenizer(prompts, padding=True, truncation=False, return_tensors='pt')
		batch["labels"] = labels
		return batch



class OwnTrainer(Trainer):
	def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		preds, lables = None, None
		eval_dataloader  = self.get_eval_dataloader(eval_dataset)
		for step, inputs in enumerate(eval_dataloader):
			input_ids = inputs['input_ids']
			with torch.no_grad():
				generated_ids = self.model.generate(input_ids=input_ids, max_new_tokens=1024)
				#generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)] #remove input from output
				compute_metrics( ( generated_ids,  inputs['labels']) )
		
		return {"accuracy": 1.0}


def compute_metrics(eval_pred):	
	generated_ids, labels = eval_pred
	pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)	
	labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	for p,l in zip(pred, labels):
		print(p, " -- LABEL:", l[-10:].replace("\n",""), "  -- gen_ids:", len(generated_ids[0]), "\n#==================\n")
	return {"accuracy": 1.0}



#==============================================================================================
if __name__ == "__main__":
	model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  #"Qwen/Qwen2-0.5B-Instruct"   #"meta-llama/Llama-3.2-1B-Instruct"  #"meta-llama/Llama-3.2-1B"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	#tokenizer.pad_token = tokenizer.eos_token #'!' #'<|finetune_right_pad_id|>' 
	#tokenizer.truncation_side = 'left'
	print("tokenizer:", tokenizer.pad_token_id, tokenizer.truncation_side)

	model = LoopedLM.from_pretrained(model_id) #LoopedLM.from_pretrained(model_id)
	#model.config.pad_token_id = tokenizer.pad_token_id

	# Dataset
	dataset = load_dataset("openai/gsm8k", "main") #  "Maxwell-Jia/AIME_2024"
	dataset = dataset.map(preprocess, batched=True)
	dataset = dataset["test"].train_test_split(test_size=0.01, seed=42)
	test_dataset = dataset["test"]
	print("Dataset test sizes:",  len(test_dataset))

	# Start training    
	data_collator = myDataCollator()
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=3,
		per_device_train_batch_size=16,
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
		#metric_for_best_model="eval_accuracy",
		remove_unused_columns=False,
		#logging_dir="./logs/",
		#report_to="tensorboard",
		#weight_decay=0.01,
	)

	trainer = OwnTrainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		#train_dataset=train_dataset,
		eval_dataset=test_dataset,
		compute_metrics=compute_metrics
	)
		
	print( trainer.evaluate() )

