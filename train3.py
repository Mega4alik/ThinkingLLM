# LLM -> SONAR emb[1024]
# venv: US1-asr3.12

import json
import os
import numpy as np
import random
import time
import datetime
import evaluate
from datasets import load_dataset, Dataset, load_from_disk
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, Qwen2ForSequenceClassification, TrainerCallback
#from utils import file_get_contents, file_put_contents
#from modeling import LoopedLM, LightweightThinkingModel
from modeling import get_averaged_layers


class Sonar:
	def __init__(self, mode):
		from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
		if mode==1:
			self.t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=torch.device("cuda"))
		else:
			self.vec2text_model = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=torch.device("cuda"))
		print("Sonar initialized")

	def encode(self, sentences):
		#sentences = ["Zachary did 46 push-ups and 58 crunches in gym class today. David did 38 more push-ups but 62 less crunches than zachary. How many more crunches than push-ups did Zachary do?", "N_02 / ( N_00 + N_01 + N_02 ) * 100.0"]
		embeddings = self.t2vec_model.predict(sentences, source_lang="eng_Latn")
		print(embeddings.shape)
		return embeddings

	def decode(self, embeddings):
		reconstructed = self.vec2text_model.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
		return reconstructed


def dataset_get_embeddings(batch):
	batch["sonar_embedding"] = sonar.encode(batch["Equation"])
	return batch


def messages_to_prompt(messages):
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def preprocess(batch): #_MAWPS sonar emb
	prompts, labels = [], []
	for i in range(len(batch["Question"])):
		messages = [
			{"role": "system", "content": "Given math question, generate equation"},
			{"role": "user", "content": batch["Question"][i]}
		]
		prompt = messages_to_prompt(messages)
		prompts.append(prompt)
		#label = str(batch["Answer"][i]).strip()+"<|im_end|>" #<|im_end|>--qwen2
		label = batch["sonar_embedding"][i]
		labels.append(label)
		#print(prompt, label); exit()
	return {"prompts":prompts, "labels":labels}


class myDataCollator:
	def __call__(self, features):
		input_ids, labels = [], []
		for sample in features:
			prompt, label = sample["prompts"], sample["labels"]            
			prompt_tokens = tokenizer(prompt, truncation=True, max_length=200).input_ids
			input_ids.append(torch.tensor(prompt_tokens))
			labels.append(label)

		input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
		labels = torch.tensor(labels)
		attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
		#print("data_collator:", input_ids.shape, labels.shape, attention_mask.shape)
		return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class MyModel(Qwen2ForSequenceClassification):
	def __init__(self, config):
		super().__init__(config)
		self.myloss = nn.MSELoss() #nn.CosineSimilarity(dim=-1)

	def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs):
		# Get standard model output (logits = sentence embedding prediction here)
		outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask) #, **kwargs
		logits = outputs.logits  # Shape: (batch_size, embedding_dim)
		if labels is not None:
			#l2_reg = torch.norm(logits, p=2, dim=-1).mean()
			loss = self.myloss(logits, labels) # + 0.01 * l2_reg
			return {"loss": loss, "logits": logits}
		else:			
			return logits
		

class OwnTrainer(Trainer):
	def predict(self, test_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		eval_dataloader  = self.get_eval_dataloader(test_dataset)
		for step, inputs in enumerate(eval_dataloader):			
			with torch.no_grad():
				preds = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
				compute_metrics( ( preds,  inputs['labels']) )
		return {"accuracy": 1.0}


def compute_metrics(p):		
	pred, labels = p
	#print(pred.dtype, labels.dtype)
	pred = sonar.decode(pred)
	labels = sonar.decode(labels)
	for p,l in zip(pred, labels):
		print(p, " -- LABEL:", l, "\n#==================\n")
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
	if False:
		sonar = Sonar(1)
		dataset = load_dataset("mwpt5/MAWPS") #"ChilleD/SVAMP"
		dataset = dataset.map(dataset_get_embeddings, batched=True, batch_size=8, num_proc=1, load_from_cache_file=False, keep_in_memory=True)
		dataset = dataset_get_embeddings(dataset)
		dataset.save_to_disk("./temp/mawps_dataset"); exit()
	else:
		dataset = load_from_disk("./temp/mawps_dataset")

	dataset = dataset.map(preprocess, batched=True)
	dataset = dataset["train"].train_test_split(test_size=0.02, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print(mode, "Dataset train, test sizes:",  len(train_dataset), len(test_dataset))
	
	# Model 	
	if False: #first training
		model = MyModel.from_pretrained(model_id, num_labels=1024)
		model.config.pad_token_id = tokenizer.pad_token_id
		# looping L=24/k(4)
		model.config.num_hidden_layers=4
		#model.model.layers = model.model.layers[:4]
		model.model.layers = get_averaged_layers(model.model.layers, 4)		
		#endOf looping		
	else:
		model = MyModel.from_pretrained("./model_temp/checkpoint-20500") #loss improving on 20500
	
	# Start training    
	data_collator = myDataCollator()
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=400,
		per_device_train_batch_size=8,
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
		weight_decay=0.01, #instead of L2(pred).mean ?
		#logging_dir="./logs/",
		#report_to="tensorboard",		
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
		trainer.train("./model_temp/checkpoint-20500")
	else:
		sonar = Sonar(2)
		print(trainer.predict(test_dataset))

