import json
import os
import sys
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

kb = json.load(open(os.path.join('../data/KQAPro.IID/', 'kb.json')))

val = json.load(open('../data/KQAPro.IID/val.json'))

def _get_direct_concepts(ent_id):
    """
    return the direct concept id of given entity/concept
    """
    if ent_id in kb['entities']:
        return kb['entities'][ent_id]['instanceOf']
    elif ent_id in kb['concepts']:
        return kb['concepts'][ent_id]['subclassOf'] # instanceOf

def _get_all_concepts(ent_id):
    """
    return a concept id list
    """
    ancestors = []
    q = Queue()
    for c in _get_direct_concepts(ent_id):
        q.put(c)
    while not q.empty():
        con_id = q.get()
        ancestors.append(con_id)
        for c in kb['concepts'][con_id]['subclassOf']:  # instaceOf
            q.put(c)
    return ancestors

entities_name_to_ids = defaultdict(list)
concepts_name_to_ids = defaultdict(list)
concept_to_entity = defaultdict(set)
concept_to_relation = {}
concept_to_attr = {}

for key in kb['entities'].keys():
    name = kb['entities'][key]['name'].lower()
    entities_name_to_ids[name].append(key)


for key in kb['concepts'].keys():
    name = kb['concepts'][key]['name'].lower()
    concepts_name_to_ids[name].append(key)

for ent_id in kb['entities']:
    for c in _get_all_concepts(ent_id): # merge entity into ancestor concepts
        concept_to_entity[c].add(ent_id)

for c in kb['concepts'].keys():
    entities_ids = concept_to_entity[c]
    unique_attributes = defaultdict(set)
    unique_relations = defaultdict(set)
    for ent_id in entities_ids:
        attributes = kb['entities'][ent_id]['attributes']
        for i in range(len(attributes)):
            unique_attributes[attributes[i]['key']].add('')
            if(kb['concepts'][c]['name'] == 'private university'):
                print("attr_kv", attributes[i])
            for qk in attributes[i]['qualifiers']:
                unique_attributes[attributes[i]['key']].add(qk)

        
        relations = kb['entities'][ent_id]['relations']
        for i in range(len(relations)):
            unique_relations[relations[i]['relation']].add('')
            for qk in relations[i]['qualifiers']:
                unique_relations[relations[i]['relation']].add(qk)
            

    concept_to_attr[c] = unique_attributes
    concept_to_relation[c] = unique_relations

        

concept_to_entity = { k:list(v) for k,v in concept_to_entity.items() }


def get_entity_concepts(idx):
    entities = []
    concepts = []
    program = val[idx]['program']
    for j in range(len(program)):
        if(program[j]['function'] == "Find"):
            entities += program[j]['inputs']
        elif(program[j]['function'] == "FilterConcept"):
            concepts += program[j]['inputs']
    entities = list(set(entities))
    concepts = list(set(concepts))
    return entities, concepts

def split_question(question, entities, concepts):
    question = question.lower()
    for ent in entities:
        question = question.replace(ent.lower(), "#SEP#")

    for con in concepts:
        question = question.replace(con.lower(), "#SEP#")

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
    # print('similarity_score', word)
    selected_words = []
    for i in range(len(similarity_score)):
        if(similarity_score[i][0] > threshold):
            selected_words.append(similarity_score[i][1])
    
    if(len(selected_words) > 0):
        print("S", selected_words[0])

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

    print("NGRAMS before filter", output)
    ## Removing stopwords
    output_after_filter = []
    for word in output:
        if word not in stop_words:
            output_after_filter.append(word)
    return output_after_filter


def retrieve_relation_hop_wise(question, entities_keys, concepts_keys, possible_relation, threshold, temp_concept_to_entity):
    entities_keys = list(set(entities_keys))
    concepts_keys = list(set(concepts_keys))

    question = question.lower()
    rels = []
    attrs = []
    qual = []
    master_list = []

    all_unique_triples = []
    next_hop_ent_keys = []
    next_hop_con_keys = []
    next_temp_concept_to_entity = {}
    ## Exploring entities
    for key in entities_keys:
        attributes = kb['entities'][key]['attributes']
        relations = kb['entities'][key]['relations']
        entity_name = kb['entities'][key]['name']
        # if(key in temp_concept_to_entity):
        #     entity_name = temp_concept_to_entity[key]
        print('Entity_name', entity_name)
        for i in range(len(attributes)):
            attributes_name = attributes[i]['key']
            print("attribute_e", attributes_name, "attr_v", attributes[i]['value'])
            # attribute_dict = {}
            # if(key in temp_concept_to_entity):
            #     attribute_dict = {'concept': entity_name,
            #                   'attribute': attributes_name}
            # else:
            attribute_dict = {'entity': entity_name,
                            'attribute': attributes_name}
            unique_triples = [entity_name, '', '', attributes_name, '']
            if(len(select_word(attributes_name, possible_relation, threshold)) > 0):
                attrs.append(attributes_name)
                for qk in attributes[i]['qualifiers']:
                    if(len(select_word(qk, possible_relation, threshold)) > 0):
                        # print(attributes[i]['qualifiers'][qk])
                        qual.append(qk)
                        attribute_dict['qualifier'] = qk
                        unique_triples[4] = qk
                if(unique_triples not in all_unique_triples):
                    master_list.append(attribute_dict)
                    all_unique_triples.append(unique_triples)
                


        for i in range(len(relations)):
            relation_name = relations[i]['relation']
            relation_key = relations[i]['object']
            relation_dict = {}
            # if(key in temp_concept_to_entity):
            #     relation_dict = {'concept': entity_name,
            #                     'relation': relation_name}
            # else:
            relation_dict = {'entity': entity_name,
                            'relation': relation_name}
            unique_triples = [entity_name, '', relation_name, '', '']
            if(len(select_word(relation_name, possible_relation, threshold)) > 0):
                rels.append(relation_name)
                if(relation_key in kb['entities'].keys()):
                    next_hop_ent_keys.append(relation_key)
                    # relation_dict['Entity_2'] = kb['entities'][relation_key]['name']
                    # unique_triples[1] = kb['entities'][relation_key]['name']
                else:
                    next_hop_con_keys.append(relation_key)
                    # relation_dict['Concept_2'] = kb['concepts'][relation_key]['name']
                    # unique_triples[1] = kb['concepts'][relation_key]['name']
                
                for qk in relations[i]['qualifiers']:
                    if(len(select_word(qk, possible_relation, threshold)) > 0):
                        # print(relations[i]['qualifiers'][qk])
                        qual.append(qk)
                        relation_dict['qualifier'] = qk
                        unique_triples[4] = qk
                
                if(unique_triples not in all_unique_triples):
                    all_unique_triples.append(unique_triples)
                    master_list.append(relation_dict)


    ## Implementing for hop1 
    for key in concepts_keys:
        relations = concept_to_relation[key]
        for relation_name, qualifiers in relations.items():
            relation_dict = {'concept': kb['concepts'][key]['name'],
                            'relation': relation_name}
            
            unique_triples = [kb['concepts'][key]['name'], '', relation_name, '', '']

            if(len(select_word(relation_name, possible_relation, threshold)) > 0):
                rels.append(relation_name)
            
                for qk in qualifiers:
                    if(len(select_word(qk, possible_relation, threshold)) > 0):
                        qual.append(qk)
                        relation_dict['qualifier'] = qk
                        unique_triples[4] = qk
                
                if(unique_triples not in all_unique_triples):
                    all_unique_triples.append(unique_triples)
                    master_list.append(relation_dict)

        attributes = concept_to_attr[key]
        for attributes_name, qualifiers in attributes.items():
            attribute_dict = {'concept': kb['concepts'][key]['name'],
                            'attribute': attributes_name}
        
            unique_triples = [kb['concepts'][key]['name'], '', '', attributes_name, '']

            if(len(select_word(attributes_name, possible_relation, threshold)) > 0):
                attrs.append(attributes_name)
                for qk in qualifiers:
                    if(len(select_word(qk, possible_relation, threshold)) > 0):
                        # print(qualifiers[qk])
                        qual.append(qk)
                        attribute_dict['qualifier'] = qk
                        unique_triples[4] = qk
                if(unique_triples not in all_unique_triples):
                    master_list.append(attribute_dict)
                    all_unique_triples.append(unique_triples)

    rels = list(set(rels))
    attrs = list(set(attrs))
    qual = list(set(qual))
    next_hop_ent_keys = list(set(next_hop_ent_keys))
    next_hop_con_keys = list(set(next_hop_con_keys))
    
    return master_list, attrs, rels, qual, next_hop_ent_keys, next_hop_con_keys, next_temp_concept_to_entity


def retrieve_relation(idx, threshold, hops):
    question = val[idx]['question']
    print("question:", question)
    entities, concepts = get_entity_concepts(idx)
    possible_relation = split_question(question, entities, concepts)
    # hops = len(possible_relation)
    # hops = 2
    possible_relation = generate_ngrams(possible_relation)
    # threshold = 0.9
    # print('possible_relation', possible_relation)
    ## Exploring entities
    entities_keys = []
    concepts_keys = []
    temp_concept_to_entity = {}

    for ent in entities:
        keys = entities_name_to_ids[ent.lower()]
        entities_keys += keys
    
    for conc in concepts:
        keys = concepts_name_to_ids[conc.lower()]
        concepts_keys += keys

    all_attrs = []
    all_relations = []
    all_qual  = []
    all_triples = []
    
    for i in range(hops):
        master_list, attrs, rels, qual, entities_keys, concepts_keys, temp_concept_to_entity = retrieve_relation_hop_wise(question, entities_keys, concepts_keys, possible_relation, threshold, temp_concept_to_entity)
        all_triples += master_list
        all_attrs += attrs
        all_relations += rels
        all_qual += qual

    all_attrs = list(set(all_attrs))
    all_relations = list(set(all_relations))
    all_qual = list(set(all_qual))

    return all_triples, all_relations, all_attrs, all_qual

def get_ground_rel_att_qual(idx):
    rel = []
    attr = []
    qual = []
    program = val[idx]['program']

    for i in range(len(program)):
        if(program[i]['function'] in ['FilterStr', 'FilterNum', 'FilterYear', 'FilterDate', 'QueryAttr', 'QueryAttrUnderCondition', 'SelectBetween', 'SelectAmong', 'QueryAttrQualifier']):
            attr.append(program[i]['inputs'][0])
        
        if(program[i]['function'] in ['QFilterStr', 'QFilterNum', 'QFilterYear', 'QFilterDate']):
            qual.append(program[i]['inputs'][0])
        elif(program[i]['function'] in ['QueryAttrUnderCondition', 'QueryRelationQualifier']):
            qual.append(program[i]['inputs'][1])
        elif(program[i]['function'] in ['QueryAttrQualifier']):
            qual.append(program[i]['inputs'][2])
        
        if(program[i]['function'] in ['Relate', 'QueryRelationQualifier']):
            rel.append(program[i]['inputs'][0])

    rel = list(set(rel))
    attr = list(set(attr))
    qual = list(set(qual))

    return rel, attr, qual     


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

p_rel, p_attr, p_qual = 0, 0, 0
r_rel, r_attr, r_qual = 0, 0, 0
val = [val[47]]

for i in tqdm(range(len(val))):
    print("idx:", i)
    try:
        all_triples, rel, attr, qual = retrieve_relation(i, threshold, hop)
    except:
        all_triples, rel, attr, qual = [], [], [], []
    print("TRIPLES:", len(all_triples))
    for j in range(len(all_triples)):
        print(all_triples[j])
    print("RELATION:", rel)
    print("ATTRIBUTE:", attr)
    print("QUALIFIERS:", qual)
    grel, gattr, gqual = get_ground_rel_att_qual(i)
    print("G_RELATION:", grel)
    print("G_ATTRIBUTE:", gattr)
    print("G_QUALIFIERS:", gqual)

    p, r = calculate_accuracy(grel, rel)
    p_rel += p
    r_rel += r

    p, r = calculate_accuracy(gattr, attr)
    p_attr += p
    r_attr += r

    p, r = calculate_accuracy(gqual, qual)
    p_qual += p
    r_qual += r
    print("---------------------------------")
    # break
    
print("Precison_rel:", p_rel/len(val))
print("Precison_attr:", p_attr/len(val))
print("Precison_qual:", p_qual/len(val))
print("Recall_rel:", r_rel/len(val))
print("Recall_attr:", r_attr/len(val))
print("Recall_qual:", r_qual/len(val))
