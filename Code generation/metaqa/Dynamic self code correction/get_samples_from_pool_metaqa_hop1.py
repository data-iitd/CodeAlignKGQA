import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
import random

hop1_samples = [19313, 3017, 17100, 14943, 14394, 17651, 15166, 16312, 1316, 16778, 13278, 37058, 4849, 40655, 37175, 4470, 6121, 8165, 12432, 10697, 38408, 10079, 28978, 27643, 29000, 22065, 27086, 24561, 33400, 28311, 24762, 22412, 28286, 34500, 34637, 34553, 34423, 34505, 34448, 34610, 34659, 34674, 34636, 34602, 45633, 45632, 45691, 45695, 45620, 45688, 45692, 45629, 45604, 45637, 45597, 48073, 46386, 47018, 48182, 46410, 46848, 47565, 46706, 47059, 46582, 46459, 48927, 57914, 55010, 51277, 53274, 56500, 52294, 48895, 56333, 58216, 57793, 75019, 83940, 78271, 83520, 82606, 77156, 85871, 76030, 82864, 82535, 61570, 93805, 73885, 95668, 95654, 68948, 74023, 95009, 73773, 65845, 67528, 65097]

model = SentenceTransformer('all-distilroberta-v1')
client = chromadb.PersistentClient(path="./")


f = open('../data/metaQA/1-hop/vanilla/qa_train.txt')
train_data = f.readlines()
f.close()

file = open('../data/metaQA/metaqa_train.json')
annotated_data = json.load(file)
file.close()

annotated_data_dict = {}
relation_dict = {}
for d in annotated_data:
    q = d['nlq']
    p = d['kopl'].replace('QueryAttr', 'RELATE').replace('Relate(', 'RELATE(').replace('Find(', 'FIND(').replace('What(', 'WHAT(').replace(',forward)', ')').replace(',backward)', ')').split(').')
    relation = p[1].replace('RELATE(', '').split(',')[0].replace(')', '')
    annotated_data_dict[q] = p

train_samples = []
for i in hop1_samples:
    train_samples.append(train_data[i].split('\t')[0].replace('[', '').replace(']', ''))

sentence_embeddings = model.encode(train_samples)

collection = client.get_or_create_collection("top-100-documents-metaqahop1-new", metadata={"hnsw:space": "cosine"})
collection.add(
        embeddings=[element.tolist() for element in sentence_embeddings],
        ids=[str(i) for i in hop1_samples], # unique for each doc
    )

idx_mapping = {}
for i in range(len(hop1_samples)):
    idx = hop1_samples[i]
    idx_mapping[idx] = i 

def get_facts(idx, facts):
    facts_str = "[\n"
    for line in range(len(facts)):
        if facts[line] == f'idx: {idx_mapping[int(idx)]}\n':
            num_triples = int(facts[line + 2].split('TRIPLES: ')[1])
            for t in range(num_triples):
                facts_str += facts[line + 3 + t][:-1] + ",\n"
            if num_triples > 0:
                facts_str = facts_str[:-2] + '\n'
                facts_str += ']\n\n'
            else:
                facts_str = '[]\n\n'
            break
    return facts_str

f1 = open("Add path of facts file for pool of hop1_samples")
facts = f1.readlines()
f1.close()


def create_icl(ques_ids):
    ques_prompt = ""
    i = 1
    inst = "Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:"
    for ques_id in ques_ids:
        ques = train_data[int(ques_id)].split('\t')[0]
        entity = ques[ques.find("[")+1:ques.find("]")]
        ques = ques.replace('[', '').replace(']', '')
        program = annotated_data_dict[ques]
        facts_str = get_facts(ques_id, facts)
        ques_prompt += f'Training Example {i}:\nquestion = "{ques}"\nentities = [\'{entity}\']\nfacts = {facts_str}{inst}\nexpression_1 = START()\n'
        for j in range(len(program)):
            if program[j].startswith('WHAT('):
                ques_prompt += f"expression_1 = {program[j][:-1]}expression_1)\n"
            elif program[j].endswith(')'):
                ques_prompt += f"expression_1 = {program[j][:-1]}, expression_1)\n"
            else:
                ques_prompt += f"expression_1 = {program[j]}, expression_1)\n"
        ques_prompt += "expression_1 = STOP(expression_1)\n\n"
        i += 1
    ques_prompt += 'Test Example:\nquestion = "'
    return ques_prompt
