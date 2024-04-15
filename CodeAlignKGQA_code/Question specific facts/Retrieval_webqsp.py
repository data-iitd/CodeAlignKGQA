import json
import os
import sys
import pandas as pd
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import torch
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from datetime import date
import re
from queue import Queue
from tqdm import tqdm
from KQAPro_Baselines.utils.value_class import ValueClass, comp, isOp

f = open("../data/webQSP/raw_kb.txt", "r")
kb = f.readlines()
f.close()

filename = "Input path of questions file"
f = open(filename, "r")
val = json.load(f)
f.close()


def processKB(kb):
    entity_name_to_ids = {}
    entity_name_to_ids2 = {}
    entities = {}
    mapping = pd.read_csv('../dataset/webQSP/mid2name.tsv',sep='\t', header=None)
    mapping2 = pd.read_csv('../dataset/webQSP/mapping_new.csv')
    
    for i in range(len(mapping)):
        id_ = '.'.join(mapping[0][i].strip('/').split('/'))
        entity_name_to_ids[mapping[1][i]] = id_
        entities[id_] = {'name': mapping[1][i], 'relations': {}, 'raw_relations': {}}
        
    for i in range(len(mapping2)):
        entity_name_to_ids2[mapping2["name"][i]] = mapping2["id_"][i]
        
    id_ = 1
    error = 0
    for line in kb:
        line = line[:-1]
        kb_elements = line.strip().split('\t')
        try:
            if kb_elements[0] not in entities.keys():
                id_ = kb_elements[0]
                entity_name_to_ids[id_] = id_
                entities[id_] = {'name': id_, 'relations': {}, 'raw_relations': {}}
            
            if kb_elements[2] not in entities.keys():
                id_ = kb_elements[2]
                entity_name_to_ids[id_] = id_
                entities[id_] = {'name': id_, 'relations': {}, 'raw_relations': {}}
            
            enitiy_id = kb_elements[0]
            if(kb_elements[1].split('.')[-1] in entities[enitiy_id]['relations']):
                entities[enitiy_id]['relations'][kb_elements[1].split('.')[-1]].append([kb_elements[2], 'forward'])
                entities[enitiy_id]['raw_relations'][kb_elements[1]].append([kb_elements[2], 'forward'])
            else:
                entities[enitiy_id]['relations'][kb_elements[1].split('.')[-1]] = [[kb_elements[2], 'forward']]
                entities[enitiy_id]['raw_relations'][kb_elements[1]] = [[kb_elements[2], 'forward']]
                    
            enitiy_id = entity_name_to_ids[kb_elements[2]]
            if(kb_elements[1].split('.')[-1] in entities[enitiy_id]['relations']):
                entities[enitiy_id]['relations'][kb_elements[1].split('.')[-1]].append([kb_elements[0], 'backward'])
                entities[enitiy_id]['raw_relations'][kb_elements[1]].append([kb_elements[0], 'backward'])
            else:
                entities[enitiy_id]['relations'][kb_elements[1].split('.')[-1]] = [[kb_elements[0], 'backward']]
                entities[enitiy_id]['raw_relations'][kb_elements[1]] = [[kb_elements[0], 'backward']]
        except:
            error = 1
    
    return entity_name_to_ids, entities, entity_name_to_ids2


entity_name_to_ids, entities, entity_name_to_ids2 = processKB(kb)

def get_entity(idx):
    program = val['Questions'][idx]
    entities = []
    
    for i in range(len(program['Parses'])):
        entities.append(program['Parses'][i]['TopicEntityName'])
    entities = list(set(entities))

    return entities

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


def retrieve_relation_hop_wise(question, entities_keys, possible_relation, threshold, key2name):
    entities_keys = list(set(entities_keys))
    # print("retrieve_rel_hop_wise")
    question = question.lower()
    rels = []
    master_list = []

    all_unique_triples = []
    next_hop_ent_keys = []

    ## Exploring entities
    for key in entities_keys:
        relations = entities[key]['relations']
        entity_name = entities[key]['name']
        if(key in key2name):
            entity_name = key2name[key]
        # print("ent_name", entity_name)
        for rel, rel_list in relations.items():
            relation_name = rel
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
    question = val['Questions'][idx]['ProcessedQuestion']
    print("question:", question)
    entities = get_entity(idx)
    # print("entities:", entities)
    possible_relation = split_question(question, entities)
    # hops = len(possible_relation)
    # hops = 2
    possible_relation = generate_ngrams(possible_relation)
    # threshold = 0.9
    # print('possible_relation', possible_relation)
    ## Exploring entities
    entities_keys = []
    temp_concept_to_entity = {}
    key2name = {}

    for ent in entities:
        # ent = ent.lower()
        if(ent in entity_name_to_ids):
            keys = entity_name_to_ids[ent]
            if(ent == keys and ent in entity_name_to_ids2):
                keys = entity_name_to_ids2[ent]

        elif(ent in entity_name_to_ids2):
            keys = entity_name_to_ids2[ent]
        else:
            keys = []

        if(isinstance(keys, list)):
            for k in keys:
                key2name[k] = ent
            entities_keys += keys
        else:
            key2name[keys] = ent
            entities_keys.append(keys)
                
    # print("keys", entities_keys)
    all_relations = []
    all_triples = []
    
    for i in range(hops):
        master_list, rels, entities_keys = retrieve_relation_hop_wise(question, entities_keys, possible_relation, threshold, key2name)
        # print("rel", rels)
        all_triples += master_list
        all_relations += rels

    all_relations = list(set(all_relations))

    return all_triples, all_relations

def extractFinalAnswer(filename):
    f = open(filename, "r")
    val = json.load(f)
    f.close()
    answers = []
    for i in range(len(val['Questions'])):
        answer = []
        for j in range(len(val['Questions'][i]['Parses'])):
            for k in range(len(val['Questions'][i]['Parses'][j]['Answers'])):
                # try:
                #     answer.append(val['Questions'][i]['Parses'][j]['Answers'][k]['EntityName'].lower())
                # except:
                answer.append(val['Questions'][i]['Parses'][j]['Answers'][k]['AnswerArgument'])
        answer = list(set(answer))
        answers.append(answer)
    return answers

def get_ground_rel(idx):
    rel = []
    program = val['Questions'][idx]
    for i in range(len(program['Parses'])):
        parses = program['Parses'][i]
        if(parses['InferentialChain'] is not None):
            for j in range(len(parses['InferentialChain'])):
                rel.append(parses['InferentialChain'][j].split('.')[-1])

    rel = list(set(rel))

    return rel


def calculate_accuracy(clean_truth, clean_pred):
    TP = 0
    FN = 0
    FP = 0

    for i in range(len(clean_pred)):
        if(clean_pred[i] in clean_truth):
            TP += 1
        else:
            FP += 1
    
    for i in range(len(clean_truth)):
        if(clean_truth[i] not in clean_pred):
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
# val = val[:5]

for i in tqdm(range(len(val['Questions']))):
    print("idx:", i)
    try:
        all_triples, rel = retrieve_relation(i, threshold, hop)
    except:
        all_triples, rel = [], []
    print("TRIPLES:", len(all_triples))
    for j in range(len(all_triples)):
        print(all_triples[j])
    print("RELATION:", rel)
    grel = get_ground_rel(i)
    print("G_RELATION:", grel)

    p, r = calculate_accuracy(grel, rel)
    p_rel += p
    r_rel += r

    print("---------------------------------")
    # break
    
print("Precison_rel:", p_rel/len(val['Questions']))
print("Recall_rel:", r_rel/len(val['Questions']))
