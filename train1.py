# reasoning through looping
# venv: US1-asr3.12

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
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
#from utils import file_get_contents, file_put_contents
#from modeling import LoopedLM, LightweightThinkingModel
from modeling import get_averaged_layers

def messages_to_prompt(messages):
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def preprocess(batch): #_svamp
	prompts, labels = [], []
	for i in range(len(batch["Question"])):
		messages = [
			{"role": "system", "content": "Given math question, generate final answer"},
			{"role": "user", "content": batch["Body"][i] + "\nQuestion: " + batch["Question"][i]}
		]
		prompt = messages_to_prompt(messages)
		prompts.append(prompt)
		label = str(batch["Answer"][i]).strip()+"<|im_end|>" #<|im_end|>--qwen2
		labels.append(label)
		print(prompt, label); exit()
	return {"prompts":prompts, "labels":labels}


class myDataCollator:
    def __call__(self, features):
        input_ids, labels = [], []
        for sample in features:
            prompt, answer = sample["prompts"], sample["labels"]

            full = f"{prompt}{answer}" # Compose full text

            full_tokens = tokenizer(full, truncation=True, max_length=280).input_ids
            prompt_tokens = tokenizer(prompt, truncation=True, max_length=200).input_ids

            label_ids = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]

            input_ids.append(torch.tensor(full_tokens if mode==1 else prompt_tokens))
            labels.append(torch.tensor(label_ids))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
        #print("data_collator:", input_ids.shape, labels.shape, attention_mask.shape)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class OwnTrainer(Trainer):
	def predict(self, test_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		eval_dataloader  = self.get_eval_dataloader(test_dataset)
		for step, inputs in enumerate(eval_dataloader):
			input_ids = inputs['input_ids']
			with torch.no_grad():
				generated_ids = self.model.generate(input_ids=input_ids, max_new_tokens=512) #, do_sample=True, num_beams=20
				generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)] #remove input from output
				compute_metrics( ( generated_ids,  inputs['labels']) )				
		return {"accuracy": 1.0}


def compute_metrics(p):	
	generated_ids, labels = p
	#labels, generated_ids = torch.tensor(p.label_ids), np.argmax(p.predictions, axis=-1) #eval
	labels[labels == -100] = tokenizer.pad_token_id
	pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
	labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	for p,l in zip(pred, labels):
		print(p, " -- LABEL:", l, "  -- gen_ids:", len(generated_ids[0]), "\n#==================\n")
	return {"eval_accuracy": 1.0}


#==============================================================================================
if __name__ == "__main__":
	mode = 1 #1-train,2-test
	model_id = "Qwen/Qwen2-0.5B-Instruct" #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # #"meta-llama/Llama-3.2-1B-Instruct"  #"meta-llama/Llama-3.2-1B"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	tokenizer.pad_token = tokenizer.eos_token #'!' #'<|finetune_right_pad_id|>' 
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.truncation_side = 'left'
	print("tokenizer:", tokenizer.pad_token_id, tokenizer.truncation_side)

	# Dataset	
	dataset = load_dataset("ChilleD/SVAMP") # "mwpt5/MAWPS"
	dataset = dataset.map(preprocess, batched=True)
	dataset = dataset["train"].train_test_split(test_size=0.02 if mode==1 else 0.1, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print(mode, "Dataset train, test sizes:",  len(train_dataset), len(test_dataset))

	# Model 	
	if True: #first training
		model = AutoModelForCausalLM.from_pretrained(model_id) #LoopedLM.from_pretrained(model_id)	
		model.config.pad_token_id = tokenizer.pad_token_id
		# looping L=24/k(4)
		model.config.num_hidden_layers=4
		#model.model.layers = model.model.layers[:4]
		model.model.layers = get_averaged_layers(model.model.layers, 4)
		#endOf looping		
	else:
		model = AutoModelForCausalLM.from_pretrained("./model_temp/checkpoint-17200")
	
	# Start training    
	data_collator = myDataCollator()
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=100,
		per_device_train_batch_size=1,
		gradient_accumulation_steps=1,
		learning_rate=1e-6,
		logging_steps=20,
		save_steps=500,
		save_total_limit=2,
		eval_strategy="steps",
		eval_steps=500,
		per_device_eval_batch_size=1,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		remove_unused_columns=False,
		#logging_dir="./logs/",
		#report_to="tensorboard",
		#weight_decay=0.01,
	)

	trainer = OwnTrainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		#compute_metrics=compute_metrics
	)
	
	if mode==1:
		trainer.train()
	else:
		print(trainer.predict(test_dataset))

