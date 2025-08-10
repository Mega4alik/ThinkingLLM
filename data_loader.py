import json, random
from datasets import load_from_disk, Dataset
from utils import file_get_contents, JinaAI

def dataset_to_dict(dataset): #specifically for train7,8: question, answers, sentences
	d = {}
	for (question, chunks, labels, answer) in dataset:
		for o in [ ("question", question), ("sentences", chunks), ("answers", [answer])]:
			k, v = o[0], o[1]
			if k not in d: d[k] = []
			d[k].append(v)
	return d


def postprocess(row):
	sentences, answers, question = row["sentences"], row["answers"], row["question"]
	row["question_emb"] = embedding_model.encode([question], isq=True)[0]#.detach().cpu() #B
	row["sentences_emb"] = embedding_model.encode(sentences)#.detach().cpu() #[ [sent, sent], ... ] #B,T
	row["answers_emb"] = embedding_model.encode(answers)#.detach().cpu() #B, T
	del row["document"]
	return row


def postprocess_batched(batch):
    sentences_list = batch["sentences"]      # List of lists of sentences (B x T)
    answers_list = batch["answers"]          # List of lists of answers (B x T)
    questions = batch["question"]            # List of strings (B)

    # Embed all questions
    batch["question_emb"] = embedding_model.encode(questions, isq=True)  # (B x D)

    # Flatten for batch encoding: sentences
    flat_sentences = [s for sublist in sentences_list for s in sublist]
    flat_answers = [a for sublist in answers_list for a in sublist]

    # Encode and regroup back into batches
    sentences_emb_flat = embedding_model.encode(flat_sentences)  # (B*T x D)
    answers_emb_flat = embedding_model.encode(flat_answers)      # (B*T x D)

    # Reshape back into nested lists of embeddings
    def regroup(flat_list, original_nested):
        output = []
        idx = 0
        for sublist in original_nested:
            output.append(flat_list[idx:idx + len(sublist)])
            idx += len(sublist)
        return output

    batch["sentences_emb"] = regroup(sentences_emb_flat, sentences_list)
    batch["answers_emb"] = regroup(answers_emb_flat, answers_list)
    return batch


def hotpotqa_prepare_data(mode): #[(question, [[chunk1, chunk2, ..]], [[label1, label2, ..]])]
	obj, dataset = json.loads(file_get_contents("/home/mega4alik/Desktop/python/rerank/data/hotpotqa/hotpot_train_v1.1.json" if mode==1 else  "/home/mega4alik/Desktop/python/rerank/data/hotpotqa/hotpot_dev_distractor_v1.json")), []	
	for q in (obj[0:] if mode==1 else obj[-100:]): #~90k all
		question, sf, answer = q["question"], q["supporting_facts"], q["answer"]
		chunks, labels = [], []
		for x in q["context"]:
			try:
				title, paragraphs = x[0], x[1]
				labels2 = [0] * len(paragraphs)
				for fact in sf:
					if fact[0]==title: labels2[fact[1]] = 1
				chunks.extend(paragraphs) #v1:chunks_list.append([..]) v2: chunks.extend([..])
				labels.extend(labels2)
			except Exception as e:
				print(e)

		dataset.append( (question, chunks, labels, answer) )
		#print(len(chunks), len(labels), question, "ans:", answer, "\n\n", json.dumps(chunks))

	return dataset

#================================================

def run1():
	dataset = hotpotqa_prepare_data(1)
	d = dataset_to_dict(dataset)
	del dataset
	mydataset = Dataset.from_dict(d)
	del d
	mydataset.save_to_disk("./temp/hotpotqa_train")


if __name__=="__main__": #run2
	embedding_model = JinaAI()
	pre = './temp'
	dataset = load_from_disk(pre+"/hotpotqa_train")
	dataset = dataset.select(range(20000, 90447))  # keep it a Dataset
	dataset = dataset.map(postprocess_batched, batched=True, batch_size=256)
	dataset.save_to_disk(pre+"/hotpotqa_train_jinaai_2")