import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
# import torch
from datetime import datetime
import json
import random

hop2_samples = [(11189, 1725), (7480, 1674), (33699, 1512), (20486, 1461), (27190, 1458), (37349, 1404), (26035, 1402), (76127, 1385), (47791, 1372), (41119, 1360), (1708, 1327), (36097, 1327), (15342, 1317), (41617, 1316), (74003, 1314), (50902, 1311), (17768, 1298), (29771, 1298), (67970, 1281), (22191, 1265), (13496, 1265), (74813, 1262), (53264, 1223), (40894, 1208), (19063, 1206), (41888, 1199), (65402, 1191), (30212, 1191), (16754, 1184), (63861, 1181), (27227, 1171), (30483, 1169), (35978, 1163), (72681, 1155), (36339, 1148), (33679, 1133), (51212, 1125), (23473, 1122), (8678, 1122), (28594, 1120), (78663, 1115), (27680, 1103), (4365, 1102), (67907, 1091), (29482, 1088), (25815, 1088), (55436, 1082), (50189, 1082), (22925, 1081), (36904, 1080), (70677, 1072), (32399, 1067), (17942, 1062), (38910, 1054), (59957, 1046), (43124, 1046), (2523, 1045), (47001, 1045), (69747, 1044), (75625, 1042), (43257, 1039), (21074, 1037), (61482, 1037), (26216, 1034), (36194, 1034), (24775, 1034), (2348, 1030), (19446, 1024), (17589, 1024), (68192, 1023), (77178, 1016), (62023, 1014), (57627, 1013), (60562, 1009), (40475, 1006), (59612, 1006), (59408, 1001), (3013, 1001), (50332, 1000), (31816, 998), (26108, 995), (71240, 995), (10420, 993), (15989, 993), (44749, 993), (68204, 991), (43176, 990), (37151, 985), (46885, 984), (17995, 983), (25817, 983), (42029, 982), (37253, 980), (60605, 979), (59197, 979), (12391, 978), (59312, 978), (31104, 977), (64579, 977), (63226, 976)]


model = SentenceTransformer('all-distilroberta-v1')
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.PersistentClient(path="./")


f = open('../data/metaQA/2-hop/vanilla/qa_train.txt')
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
for i,_ in hop2_samples:
    train_samples.append(train_data[i].split('\t')[0].replace('[', '').replace(']', ''))

sentence_embeddings = model.encode(train_samples)

collection = client.get_or_create_collection("top-100-documents-metaqahop2-new", metadata={"hnsw:space": "cosine"})
collection.add(
        embeddings=[element.tolist() for element in sentence_embeddings],
        ids=[str(i) for i,_ in hop2_samples], # unique for each doc
    )

idx_mapping = {}
for i in range(len(hop2_samples)):
    idx, _ = hop2_samples[i]
    idx_mapping[idx] = i 


def get_facts(idx, facts):
    facts_str = "[\n"
    for line in range(len(facts)):
        if facts[line] == f'idx: {idx_mapping[int(idx)]}\n':
            # print("facts", facts[line+3])
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

f1 = open("Add path of facts file for pool of hop2_samples")
facts = f1.readlines()
# print(facts)
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
