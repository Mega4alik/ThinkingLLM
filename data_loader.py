import json, random
from utils import file_get_contents
from datasets import Dataset

def dataset_to_dict(dataset): #specifically for train7,8: question, answers, sentences
	d = {}
	for (question, chunks, labels, answer) in dataset:
		for o in [ ("question", question), ("sentences", chunks), ("answers", [answer])]:
			k, v = o[0], o[1]
			if k not in d: d[k] = []
			d[k].append(v)
	return d



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


if __name__=="__main__":
	dataset = hotpotqa_prepare_data(1)
	d = dataset_to_dict(dataset)
	del dataset
	mydataset = Dataset.from_dict(d)
	del d
	mydataset.save_to_disk("./temp/hotpotqa_train")