import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json

top_100_samples = [1894, 2447, 1287, 1776, 174, 974, 133, 347, 2227, 1296, 250, 2782, 68, 2663, 774, 1315, 2387, 1294, 2546, 2993, 1956, 2656, 2260, 388, 1607, 1047, 2774, 2646, 3039, 1107, 2431, 726, 1609, 828, 1622, 1679, 1748, 2284, 43, 1226, 606, 2094, 2080, 2767, 2505, 513, 1109, 676, 1387, 1716, 985, 1591, 1075, 1492, 1699, 1088, 2392, 921, 716, 1164, 1926, 2126, 2270, 890, 2811, 801, 162, 1142, 1721, 87, 2620, 2879, 1909, 1563, 1360, 933, 2149, 2438, 1743, 2402, 2148, 932, 2012, 2594, 49, 2308, 2955, 124, 805, 1050, 2140, 818, 2286, 1640, 2007, 1528, 2334, 2709, 983, 1407]

model = SentenceTransformer('all-distilroberta-v1')
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.PersistentClient(path="./")

f = open("../data/webqsp/WebQSP.train.json")
train_data = json.load(f)['Questions']
f.close()


f = open("Add path of facts file for top_100_samples")
facts = f.readlines()
f.close()

train_samples = [train_data[i]['RawQuestion'] for i in top_100_samples]
sentence_embeddings = model.encode(train_samples)
collection = client.get_or_create_collection("top-100-documents-webqsp_new", metadata={"hnsw:space": "cosine"})
collection.add(
        embeddings=[element.tolist() for element in sentence_embeddings],
        ids=[str(i) for i in top_100_samples], # unique for each doc
    )

def get_facts(ques_id):
    facts_str = "[\n"
    idx = top_100_samples.index(int(ques_id))
    for line in range(len(facts)):
        if facts[line] == f'idx: {idx}\n':
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

def create_icl(ques_ids, train_data):
    ques_prompt = ""
    i = 1
    inst = "Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:"
    for ques_id in ques_ids:
        # print(len(train_data))
        ques = train_data[int(ques_id)]['RawQuestion']
        entity = train_data[int(ques_id)]['Parses'][0]['TopicEntityName']
        chain = train_data[int(ques_id)]['Parses'][0]['InferentialChain']
        constraints = train_data[int(ques_id)]['Parses'][0]['Constraints']
        facts_str = get_facts(ques_id)
        if chain:
            ques_prompt += f'Training Example {i}:\nquestion = "{ques}"\nentities = [\'{entity}\']\nfacts = {facts_str}{inst}\nexpression_1 = START()\nexpression_1 = FIND({entity}, expression_1)\n'
            for j in range(len(chain)):
                relation = chain[j].split('.')[-1]
                if relation.endswith('_s'):
                    ques_prompt += f"expression_1 = RELATE({''.join(relation.split('_'))}, expression_1)\n"
                else:
                    ques_prompt += f"expression_1 = RELATE({' '.join(relation.split('_'))}, expression_1)\n"

            if constraints:
                for k in range(len(constraints)):
                    relation = constraints[k]['NodePredicate'].split('.')[-1]
                    entityname = constraints[k]['EntityName']
                    if relation != "from" and relation != "to":
                        ques_prompt += f"expression_1 = FILTERSTR({' '.join(relation.split('_'))}, {entityname}, expression_1)\n"
            ques_prompt += "expression_1 = STOP(expression_1)\n\n"
            i += 1
    ques_prompt += 'Test Example:\nquestion = "'
    return ques_prompt
