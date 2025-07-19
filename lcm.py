# [SONAR emb[1024], ..] -> SONAR emb[1024]
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
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, TrainerCallback
from transformers import Qwen2Model
#from utils import file_get_contents, file_put_contents
#from modeling import LoopedLM, LightweightThinkingModel
from modeling import get_averaged_layers


class NLP:
	def __init__(self):
		import spacy
		self.nlp = spacy.load("en_core_web_sm")

	def split_sentences(self, text):
		doc = self.nlp(text)
		sentences = [sent.text.strip() for sent in doc.sents]
		return sentences


class Sonar:
	def __init__(self, mode):
		from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
		if mode==1 or mode==3:
			self.t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=torch.device("cuda"))
		if mode==2 or mode==3:
			self.vec2text_model = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=torch.device("cuda"))
		print("Sonar initialized")

	def encode(self, sentences):
		#sentences = ["Zachary did 46 push-ups and 58 crunches in gym class today. David did 38 more push-ups but 62 less crunches than zachary. How many more crunches than push-ups did Zachary do?", "N_02 / ( N_00 + N_01 + N_02 ) * 100.0"]
		embeddings = self.t2vec_model.predict(sentences, source_lang="eng_Latn")		
		return embeddings

	def decode(self, embeddings):
		reconstructed = self.vec2text_model.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
		return reconstructed


def preprocess(batch):
	sentences, answers, questions = [], [], []
	for i in range(len(batch["question"])):
		question = batch["question"][i]["text"]
		anss = [x["text"] for x in batch["answers"][i]]
		text = batch["document"][i]["summary"]["text"]
		sents = nlp.split_sentences(text)
		#print(text, sentences, len(text), len(sentences)); exit()
		questions.append(question)
		sentences.append(sents)
		answers.append(anss)
	return {"sentences":sentences, "answers":answers, "question":questions}


def postprocess(row):
	sentences, answers, question = row["sentences"], row["answers"], row["question"]
	row["question_emb"] = sonar.encode([question])[0].detach().cpu() #B
	row["sentences_emb"] = sonar.encode(sentences).detach().cpu() #[ [sent, sent], ... ] #B,T
	row["answers_emb"] = sonar.encode(answers).detach().cpu() #B, T
	del row["document"]
	return row


class myDataCollator:
	def __init__(self):
		self.pad_value = 0.0

	def __call__(self, batch):
		input_embs = [ torch.tensor([x['question_emb']] + x['sentences_emb']) for x in batch]  # List of [num_sent, emb_dim] tensors
		target_embs = [ torch.tensor(random.choice(x['answers_emb'])) for x in batch]  # List of [emb_dim] tensors

		padded_inputs = pad_sequence(input_embs, batch_first=True, padding_value=self.pad_value)

		# Attention mask: 1 for real sentence embeddings, 0 for padding
		attention_mask = torch.tensor([
			[1]*emb.size(0) + [0]*(padded_inputs.size(1) - emb.size(0)) for emb in input_embs
		], dtype=torch.long)

		# Stack target embeddings
		target_embs = torch.stack(target_embs)  # [batch_size, emb_dim]

		return {
			'inputs_embeds': padded_inputs,      # [B, T, C]
			'attention_mask': attention_mask,    # [B, T]
			'target_embeds': target_embs         # [B, C]
		}


class MyModel(Qwen2Model):
	def __init__(self, config):
		super().__init__(config)
		self.embedding_dim = 1024 #sonar:1024
		self.hidden_dim = 896 #qwen:896
		#self.ln1 = nn.LayerNorm(self.embedding_dim)
		self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
		self.fc2 = nn.Linear(self.hidden_dim, self.embedding_dim)
	
	def trans(self, x):
		#print("Input has NaNs, Max abs value:", torch.isnan(x).any(), x.abs().max())		
		#print("fc1 weight or bias has NaNs:", torch.isnan(self.fc1.weight).any(), torch.isnan(self.fc1.bias).any())		
		#x = self.ln1(x) #loss close to 0
		x = self.fc1(x)
		#print("trans.2", x[0], x.dtype, torch.isnan(x).any())		
		return x

	def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, target_embeds=None, **kwargs):
		outputs = super().forward(inputs_embeds=self.trans(inputs_embeds), attention_mask=attention_mask) #, **kwargs
		last_hidden_state = outputs.last_hidden_state[:,-1] #B,C
		#print("forward:", last_hidden_state[0])
		logits = self.fc2(last_hidden_state) #B,embedding_dim		
		if target_embeds is not None:
			loss = myloss(logits, target_embeds)
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
	model_id = "Qwen/Qwen2-0.5B-Instruct" #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" | "meta-llama/Llama-3.2-1B-Instruct" | "meta-llama/Llama-3.2-1B"
	#tokenizer = AutoTokenizer.from_pretrained(model_id)
	
	# Dataset
	if False:
		sonar = Sonar(1)
		nlp = NLP()
		dataset = load_dataset("deepmind/narrativeqa", split="validation") #"mwpt5/MAWPS" | "ChilleD/SVAMP"
		dataset = dataset.map(preprocess, batched=True, batch_size=8, num_proc=1, load_from_cache_file=False, keep_in_memory=True)		
		#dataset = postprocess(dataset)
		dataset = dataset.map(postprocess)
		dataset.save_to_disk("./temp/narrativeqa_dataset"); exit()
	else:
		dataset = load_from_disk("./temp/narrativeqa_dataset")
	
	dataset = dataset.train_test_split(test_size=0.02, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print(mode, "Dataset train, test sizes:",  len(train_dataset), len(test_dataset))
	
	# Model
	if True: #first training
		model = MyModel.from_pretrained(model_id) #, torch_dtype=torch.float16
		model.config.num_hidden_layers=12
		model.layers = get_averaged_layers(model.layers, 12) #model.model.layers[:4]
	else:
		model = MyModel.from_pretrained("./model_temp/checkpoint-")
	
	# Start training    
	myloss = nn.MSELoss()
	data_collator = myDataCollator()
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=100,
		per_device_train_batch_size=16,
		gradient_accumulation_steps=1,
		learning_rate=1e-6,
		logging_steps=20,
		save_steps=5000,
		save_total_limit=2,
		eval_strategy="steps",
		eval_steps=5000,
		per_device_eval_batch_size=1,
		#metric_for_best_model="eval_loss",
		greater_is_better=False,
		remove_unused_columns=False,
		weight_decay=0.01, #instead of L2(pred).mean ?
		warmup_steps=10000,
		#fp16=True,
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
		trainer.train() #"./model_temp/checkpoint-150"
	else:
		sonar = Sonar(2)
		print(trainer.predict(test_dataset))

