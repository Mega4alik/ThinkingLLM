# like lcm.py check for duplicates!
# [semb1, semb2, t1, t2, t3] -> [t2,t3,t4]
# venv: US1-asr3.12

import json, os, random, time
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, TrainerCallback
from transformers import Qwen2Model,Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from utils import Sonar, NLP, JinaAI, get_magnitudes
from modeling import get_averaged_layers, freeze_some_layers

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
	row["question_emb"] = embedding_model.encode([question], isq=True)[0]#.detach().cpu() #B
	row["sentences_emb"] = embedding_model.encode(sentences)#.detach().cpu() #[ [sent, sent], ... ] #B,T
	row["answers_emb"] = embedding_model.encode(answers)#.detach().cpu() #B, T
	del row["document"]
	return row


def find_subsequence(sequence, subseq):
	subseq = torch.tensor(subseq)
	for i in range(len(sequence) - len(subseq) + 1):
		if torch.equal(sequence[i:i+len(subseq)], subseq):
			return i
	return -1


class FTDataCollator:
	def __init__(self):
		self.pad_value = 0.0

	def apply_template(self, question, answer):
		return f"<|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n" + (f"{answer}<|im_end|>" if mode==1 else "") #<|endoftext|>?

	def __call__(self, batch):
		# sentences
		inputs_embeds = [torch.tensor(x['sentences_emb']) for x in batch] # + [x['question_emb']]
		att_s = [ torch.ones(x.size(0))  for x in inputs_embeds]
		att_s = pad_sequence(att_s, batch_first=True, padding_value=0)
		att_s = att_s.ne(0) #B,S
		inputs_embeds = pad_sequence(inputs_embeds, batch_first=True, padding_value=self.pad_value) #B,S,C
		
		# tokens
		st = [self.apply_template(x['question'], random.choice(x['answers'])) for x in batch]
		x = tokenizer(st, return_tensors="pt", padding=True, return_attention_mask=True)
		token_ids, att_t = x.input_ids, x.attention_mask
		
		# labels
		labels = []
		for t_ids in token_ids:
			answer_start_id = find_subsequence(t_ids, [77091, 198]) #assistant\n
			#label = [-100] * S + t_ids[1:] + [-100] * (pad_len + 1)  # shift left          
			label = [-100] * inputs_embeds.size(1) + [-100] * (answer_start_id+1) + [-100 if v==151643 else v for v in t_ids[answer_start_id+1+1:].tolist()] + [-100] #padding of tokens!!! stupid
			#print(t_ids, label[inputs_embeds.size(1):], att_t[len(labels)])
			labels.append(label)
		labels = torch.tensor(labels, dtype=torch.long)

		#print("inputs_embeds", inputs_embeds, "\ntoken_ids", token_ids,"\natt_t", att_t, "\nlabels", labels); exit()
		out = {'inputs_embeds': inputs_embeds, 'token_ids': token_ids, 'att_s':att_s, 'att_t':att_t, 'labels':labels}
		if mode==2: #test
			out["question"] = [x["question"] for x in batch]
			out["answers"] = [x["answers"] for x in batch]
		return out


class PTDataCollator:
	def __init__(self):
		self.pad_value = 0.0

	def apply_template(self, sentences):
		return f"<|im_start|>" + (f"{" ".join(sentences)}<|im_end|>" if mode==1 else "")

	def __call__(self, batch):		
		for x in batch: x["idx"] = random.randrange( len(x["sentences"])-1)

		inputs_embeds = [torch.tensor( x['sentences_emb'][x['idx']:x['idx']+2] ) for x in batch] # + [x['question_emb']]
		att_s = [ torch.ones(x.size(0))  for x in inputs_embeds]
		att_s = pad_sequence(att_s, batch_first=True, padding_value=0)
		att_s = att_s.ne(0) #B,S
		inputs_embeds = pad_sequence(inputs_embeds, batch_first=True, padding_value=self.pad_value) #B,S,C
		
		# tokens
		st = [ self.apply_template(x['sentences'][x['idx']:x['idx']+2]) for x in batch]
		x = tokenizer(st, return_tensors="pt", padding=True, truncation=True, max_length=40, return_attention_mask=True)
		token_ids, att_t = x.input_ids, x.attention_mask
		
		# labels
		labels = []
		for t_ids in token_ids:
			label = [-100] * inputs_embeds.size(1) +  [-100 if v==tokenizer.pad_token_id else v for v in t_ids[1:].tolist()] + [-100]
			#print(t_ids, label[inputs_embeds.size(1):], att_t[len(labels)], "\n")
			labels.append(label)
		labels = torch.tensor(labels, dtype=torch.long)

		#print("inputs_embeds", inputs_embeds, "\ntoken_ids", token_ids,"\natt_t", att_t, "\nlabels", labels); exit()
		out = {'inputs_embeds': inputs_embeds, 'token_ids': token_ids, 'att_s':att_s, 'att_t':att_t, 'labels':labels}
		return out



class MyModel(Qwen2ForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.embedding_dim = 1024 #sonar:1024
		self.hidden_dim = 896 #qwen:896
		self.embed_types = nn.Embedding(2, self.hidden_dim) #number of types, hidden_dim
		#self.ln1 = Qwen2RMSNorm(self.embedding_dim) #nn.LayerNorm
		#self.ln2 = Qwen2RMSNorm(self.hidden_dim)
		self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
	
	def trans(self, inputs_embeds, token_ids):
		tokens_embeds = self.model.embed_tokens(token_ids)
		if inputs_embeds is not None:
			x = self.fc1(inputs_embeds)
			x = torch.cat([x, tokens_embeds], dim=1)
			B, S, T = inputs_embeds.size(0), inputs_embeds.size(1), token_ids.size(1)
			type_ids = torch.cat([torch.zeros((B, S), dtype=torch.long), torch.ones((B, T), dtype=torch.long)], dim=1).to(device)
		else: #not considering this case yet
			pass
			#x = tokens_embeds
			#type_ids = torch.ones((tokens_embeds.size(0), tokens_embeds.size(1)), dtype=torch.long)

		type_embeds = self.embed_types(type_ids)
		x = x + type_embeds
		#print("\nmagnitudes:", get_magnitudes([inputs_embeds, tokens_embeds, type_embeds]));exit()
		return x

	def forward(self, inputs_embeds=None, token_ids=None, att_s=None, att_t=None, labels=None, **kwargs):
		#print("\n\nforward. inputs_embeds", inputs_embeds, "\ntoken_ids", token_ids, "\natt_s", att_s, "\natt_t", att_t)
		attention_mask = torch.cat([att_s, att_t], dim=1)   #B,S+T,C
		outputs = super().forward(inputs_embeds=self.trans(inputs_embeds, token_ids), attention_mask=attention_mask, labels=labels) #, **kwargs
		return outputs

	def generate(self, inputs_embeds=None, token_ids=None, att_s=None, max_new_tokens=32): #primitive version without caching       
		generated = []
		for i in range(max_new_tokens):
			att_t = torch.tensor( [[True] * token_ids.size(1)], device=device)
			attention_mask = torch.cat([att_s, att_t], dim=1).to(device) #B,S+T,C
			outputs = super().forward(inputs_embeds=self.trans(inputs_embeds, token_ids), attention_mask=attention_mask, use_cache=True)
			#past_key_values = outputs.past_key_values
			logits = outputs.logits[:, -1, :]
			next_token = torch.argmax(logits, dim=-1, keepdim=True)
			generated.append(next_token)
			if next_token.item() == tokenizer.eos_token_id: break #15164
			token_ids = torch.cat([token_ids, next_token], dim=1)
		
		return torch.cat(generated, dim=-1)


	def generate2(self, inputs_embeds=None, token_ids=None, att_s=None, max_new_tokens=32):
		#attention_mask = att_t     
		for i in range(max_new_tokens):
			outputs = self.forward(
				token_ids=cur_token,
				att_t=att_t,
				past_key_values=past_key_values,
				use_cache=True
			)
			logits = outputs.logits[:, -1, :]
			past_key_values = outputs.past_key_values
			next_token = torch.argmax(logits, dim=-1, keepdim=True)
			generated.append(next_token)
			cur_token = next_token
			if next_token.item() == tokenizer.eos_token_id: break #15164
		gen_ids = torch.cat(generated, dim=-1)
		return gen_ids


class OwnTrainer(Trainer):
	def predict(self, test_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		eval_dataloader  = self.get_eval_dataloader(test_dataset)
		for step, inputs in enumerate(eval_dataloader):
			with torch.no_grad():
				gen_ids = self.model.generate(inputs_embeds=inputs['inputs_embeds'], token_ids=inputs['token_ids'], att_s=inputs['att_s'])              
				generated = tokenizer.batch_decode(gen_ids)
				print(inputs['question'], "-----", generated, "ans:", inputs['answers'], "\n\n")
				#compute_metrics( (preds, inputs['answers'], inputs['question']) )
		return {"accuracy": 1.0}


def compute_metrics(p):
	pred, answers, question = p
	pred = sonar.decode(pred)
	for p, ans, ques in zip(pred, answers, question):
		print(p, " -- q", ques, "ans:", ans, "\n#==================\n")
	return {"eval_accuracy": 1.0}


#==============================================================================================
if __name__ == "__main__":
	device = torch.device("cuda")
	mode = 1 #1-train,2-test
	model_id = "Qwen/Qwen2-0.5B" #Qwen/Qwen2-0.5B | Qwen/Qwen2-0.5B-Instruct | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | "meta-llama/Llama-3.2-1B-Instruct" | "meta-llama/Llama-3.2-1B"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	
	# Dataset
	if False:
		embedding_model = JinaAI()  #Sonar(1)
		nlp = NLP()
		dataset = load_dataset("deepmind/narrativeqa", split="train[20000:]") #"mwpt5/MAWPS" | "ChilleD/SVAMP"
		dataset = dataset.map(preprocess, batched=True, batch_size=12)
		print("preprocess finished")
		dataset = dataset.map(postprocess)
		dataset.save_to_disk("./temp/narrativeqa_train_20k_2")
		exit()
	else:
		#dataset = load_from_disk("./temp/narrativeqa_train_20k")
		dataset = concatenate_datasets([ load_from_disk(fname) for fname in ["./temp/narrativeqa_train_20k", "./temp/narrativeqa_train_20k_2"] ])
	
	dataset = dataset.train_test_split(test_size=0.005, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print(mode, "Dataset train, test sizes:",  len(train_dataset), len(test_dataset))
	
	# Model
	if mode==1: #first training
		model = MyModel.from_pretrained(model_id) #, torch_dtype=torch.float16
		#model.config.num_hidden_layers=12
		#model.model.layers = get_averaged_layers(model.model.layers, 12) #model.model.layers[:4]
		freeze_some_layers(model.model.layers, 0, 6)
		#print(model, model.config); exit()
	else:
		model = MyModel.from_pretrained("./model_temp/checkpoint-")
	
	# Start training
	print("starting", "Traininig" if mode==1 else "Testing")
	data_collator = PTDataCollator() #FT/PT
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=5,
		per_device_train_batch_size=4,
		gradient_accumulation_steps=1,
		learning_rate=1e-6,
		logging_steps=20,
		save_steps=5000,
		save_total_limit=2,
		eval_strategy="steps",
		eval_steps=5000,
		per_device_eval_batch_size=1,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		remove_unused_columns=False,
		weight_decay=0.01, #instead of L2(pred).mean ?
		#warmup_steps=10000,
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
		trainer.train() #"./model_temp/checkpoint-10000"
	else:
		print(trainer.predict(test_dataset))

