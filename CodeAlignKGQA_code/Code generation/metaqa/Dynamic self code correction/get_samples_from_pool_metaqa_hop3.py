import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
# import torch
from datetime import datetime
import json
import random

hop3_samples = [(26841, 3473), (8007, 3096), (25624, 3071), (21428, 2913), (19492, 2730), (5129, 2609), (14015, 2597), (31691, 2585), (59886, 2576), (63987, 2514), (54465, 2468), (31823, 2451), (55753, 2444), (3291, 2276), (47071, 2265), (68588, 2214), (34957, 2180), (61106, 2175), (8626, 2173), (39737, 2170), (5993, 2165), (58319, 2103), (22808, 2102), (28902, 2097), (30525, 2094), (55537, 2055), (35616, 2051), (58548, 2030), (54813, 2015), (39088, 1994), (72900, 1984), (37466, 1951), (32538, 1906), (40817, 1895), (4353, 1874), (46385, 1871), (33754, 1852), (76200, 1845), (10532, 1844), (24307, 1842), (10478, 1835), (33675, 1819), (50244, 1815), (64453, 1813), (44897, 1799), (63166, 1785), (8339, 1768), (32573, 1752), (1453, 1752), (12227, 1745), (47533, 1726), (71706, 1723), (26626, 1720), (55693, 1718), (37928, 1717), (369, 1717), (62672, 1715), (55822, 1715), (76710, 1714), (74494, 1714), (4025, 1705), (76402, 1702), (33363, 1701), (43116, 1697), (17848, 1692), (58316, 1678), (72150, 1674), (41329, 1671), (59812, 1659), (10987, 1657), (46330, 1652), (41879, 1631), (38315, 1627), (47094, 1625), (76821, 1611), (497, 1606), (74679, 1601), (4276, 1594), (67138, 1593), (60891, 1585), (22560, 1583), (2789, 1582), (64238, 1578), (52193, 1577), (57660, 1571), (63337, 1571), (39798, 1563), (11013, 1562), (72172, 1550), (63392, 1545), (3382, 1539), (59263, 1533), (41161, 1532), (18573, 1531), (51615, 1530), (43869, 1529), (64386, 1528), (67806, 1524), (19921, 1522), (37723, 1519)]

model = SentenceTransformer('all-distilroberta-v1')
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.PersistentClient(path="./")

f = open('../data/metaQA/3-hop/vanilla/qa_test.txt')
test_data = f.readlines()
f.close()

f = open('../data/metaQA/3-hop/vanilla/qa_train.txt')
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
for i,_ in hop3_samples:
    train_samples.append(train_data[i].split('\t')[0].replace('[', '').replace(']', ''))


sentence_embeddings = model.encode(train_samples)
collection = client.get_or_create_collection("top-100-documents-metaqahop3-new", metadata={"hnsw:space": "cosine"})
collection.add(
        embeddings=[element.tolist() for element in sentence_embeddings],
        ids=[str(i) for i,_ in hop3_samples], # unique for each doc
    )

idx_mapping = {}
for i in range(len(hop3_samples)):
    idx, _ = hop3_samples[i]
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

f1 = open("Add path of facts file for pool of hop3_samples")
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
