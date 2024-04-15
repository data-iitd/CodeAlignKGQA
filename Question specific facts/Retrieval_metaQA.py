import json
import sys
import pandas as pd
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from tqdm import tqdm

f = open("../data/metaQA/kb.txt")
kb = f.readlines()
f.close()

filename =  sys.argv[3] # Path of file for facts retrieval 
f = open(filename, "r")
val = f.readlines()
f.close()


def processKB(kb):
    error = 0
    entities = {}
    for line in kb:
        kb_elements = line.strip().split('|')

        if kb_elements[0] not in entities.keys():
            id_ = kb_elements[0]
            entities[id_] = {'name': id_, 'relations': {}}
        
        if kb_elements[2] not in entities.keys():
            id_ = kb_elements[2]
            entities[id_] = {'name': id_, 'relations': {}}

        entity_id = kb_elements[0]
        if(kb_elements[1] in entities[entity_id]['relations']):
            entities[entity_id]['relations'][kb_elements[1]].append([kb_elements[2], 'forward'])
        else:
            entities[entity_id]['relations'][kb_elements[1]] = [[kb_elements[2], 'forward']]
        
        entity_id = kb_elements[2]
        if(kb_elements[1] in entities[entity_id]['relations']):
            entities[entity_id]['relations'][kb_elements[1]].append([kb_elements[0], 'backward'])
        else:
            entities[entity_id]['relations'][kb_elements[1]] = [[kb_elements[0], 'backward']]

    return entities
entities = processKB(kb)

def split_question(question, entities):
    question = question.lower()
    for ent in entities:
        question = question.replace(ent.lower(), "#SEP#")

    question = question.split("#SEP#")
    return question

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_bert = SentenceTransformer('all-distilroberta-v1').to(device)

def calculate_cosine_similarity_opt_new(currentQuery, queryListFinal):
    embeddings = model_bert.encode(queryListFinal)
    currentQueryEmbeedings = model_bert.encode(currentQuery)
    similarities = cosine_similarity(torch.from_numpy(currentQueryEmbeedings), torch.from_numpy(embeddings))
    closest = similarities.argsort(descending=True)
    sorted_list = []
    for ind in closest[:10]:
        sorted_list.append((similarities[ind].item(), queryListFinal[ind]))
    return sorted_list

def select_word(word, word_list, threshold):
    word = word.lower()
    word_list = list(set(word_list))
    if(word in word_list):
        return word
    similarity_score = calculate_cosine_similarity_opt_new(word, word_list)
    # print('similarity_score', similarity_score)
    selected_words = []
    for i in range(len(similarity_score)):
        if(similarity_score[i][0] > threshold):
            selected_words.append(similarity_score[i][1])

    return selected_words

stop_words = set(stopwords.words('english'))
def generate_ngrams(sampleList, max_size=3):
    output = []
    for sampleText in sampleList:
        # sampleText = re.sub(r'[^\w\s]', '', sampleText)
        for i in range(max_size):
            NGRAMS=ngrams(sequence=nltk.word_tokenize(sampleText), n=i+1)
            for grams in NGRAMS:
                output.append(' '.join(grams))
    ## Removing stopwords
    output_after_filter = []
    for word in output:
        if word not in stop_words:
            output_after_filter.append(word)
    return output_after_filter


def retrieve_relation_hop_wise(question, entities_keys, possible_relation, threshold):
    entities_keys = list(set(entities_keys))
    # print("entities_keys", entities_keys)
    question = question.lower()
    rels = []
    master_list = []

    all_unique_triples = []
    next_hop_ent_keys = []

    ## Exploring entities
    for key in entities_keys:
        relations = entities[key]['relations']
        # print('relations', relations)
        entity_name = entities[key]['name']
    
        # print("ent_name", entity_name)
        for rel, rel_list in relations.items():
            relation_name = rel
            # print("relation_name", relation_name)
            # print("possible", possible_relation)
            relation_keys = rel_list
            relation_dict = {'entity': entity_name,
                             'relation': relation_name}
            unique_triples = [entity_name, '', relation_name, '', '']
            # print("rel_name", relation_name)

            if(len(select_word(relation_name, possible_relation, threshold)) > 0):
                # print("selected_rel", relation_name)
                rels.append(relation_name)
                relation_key = []
                for rkey in relation_keys:
                    relation_key.append(rkey[0])
                next_hop_ent_keys += relation_key

                if(unique_triples not in all_unique_triples):
                    all_unique_triples.append(unique_triples)
                    master_list.append(relation_dict)

    rels = list(set(rels))
    next_hop_ent_keys = list(set(next_hop_ent_keys))
    
    return master_list, rels, next_hop_ent_keys

def retrieve_relation(idx, threshold, hops):
    # question = val[idx]['nlq']
    question = val[idx].split('\t')[0]
    entities = [question[question.find("[")+1:question.find("]")]]
    question = question.replace('[', '').replace(']', '')
    print("question:", question)
    # entities = get_entity(idx)
    
    # print("entities:", entities)
    possible_relation = split_question(question, entities)
    # hops = len(possible_relation)
    # hops = 2
    possible_relation = generate_ngrams(possible_relation)
    # threshold = 0.9
    # print('possible_relation', possible_relation)
    ## Exploring entities
    entities_keys = entities
    temp_concept_to_entity = {}
                
    # print("keys", entities_keys)
    all_relations = []
    all_triples = []
    
    for i in range(hops):
        master_list, rels, entities_keys = retrieve_relation_hop_wise(question, entities_keys, possible_relation, threshold)
        # print("rel", rels)
        all_triples += master_list
        all_relations += rels

    all_relations = list(set(all_relations))

    return all_triples, all_relations

def calculate_accuracy(clean_truth, clean_pred):
    TP = 0
    FN = 0
    FP = 0
    
    clean_pred_mod = []
    for i in range(len(clean_pred)):
        clean_pred_mod.append(' '.join(clean_pred[i].split('_')))

    for i in range(len(clean_pred)):
        if(clean_pred[i] in clean_truth or clean_pred_mod[i] in clean_truth):
            TP += 1
        else:
            FP += 1
    
    for i in range(len(clean_truth)):
        if(clean_truth[i] not in clean_pred and clean_truth[i] not in clean_pred_mod):
            FN += 1

    p = 0
    r = 0
    if(TP+FP == 0):
        p = 1    
    else:
        p = TP/(TP+FP)

    if(TP+FN == 0):
        r = 1
    else:
        r = TP/(TP+FN)

    return p, r

threshold = float(sys.argv[1])
hop = int(sys.argv[2])
print("threshold:", threshold)
print("hop:", hop)

p_rel = 0
r_rel = 0
# len(val)
for i in tqdm(range(0,len(val))):
    print("idx:", i)
    try:
        all_triples, rel = retrieve_relation(i, threshold, hop)
    except:
        all_triples, rel = [], []
    print("TRIPLES:", len(all_triples))
    for j in range(len(all_triples)):
        print(all_triples[j])
    print("RELATION:", rel)

    print("---------------------------------")
    # break
