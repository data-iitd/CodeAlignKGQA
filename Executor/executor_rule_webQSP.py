from utils.value_class import ValueClass, comp, isOp
from utils.misc import invert_dict
from utils.find_closest_webqsp import nearest_neigh, nearest_kv, nearest_neigh_by_id, calculate_cosine_similarity_opt_new
from Program.executor_abalation_functinal_new import RuleExecutorFunctional
import os
import sys
import json
from collections import defaultdict
from datetime import date
from queue import Queue
from tqdm import tqdm
import pandas as pd

class RuleExecutor(object):
    def __init__(self, entities, entity_name_to_ids, entity_name_to_ids2):
        # print('load kb')
        self.entities = entities
        self.idx = -1
        self.end = False
        self.lastEntitySet = []

        # replace adjacent space and tab in name
        for ent_id, ent_info in self.entities.items():
            ent_info['name'] = ' '.join(ent_info['name'].split())

        self.entity_name_to_ids = entity_name_to_ids
        self.entity_name_to_ids2 = entity_name_to_ids2

    def forward(self, clean_program, idx, 
                ignore_error=False, show_details=False):
        self.idx = idx
        self.end = False
        self.lastEntitySet = []
        self.lastFactDict = {}
        self.lastEntityDict = {}
        expressions = {}
        res = ""

        try:
            count_step = 0        
            for exp, p, inp in clean_program:
                if (count_step == len(clean_program)-2):
                    self.end = True
                count_step += 1
                if p == 'START':
                    expressions[exp] = []
                    continue
                elif p == 'STOP':
                    expressions[exp] = res
                    break
                else:
                    if(p == "QUERYATTR"):
                        p = "RELATE"
                    print(p)
                    func = getattr(self, p)
                    res = []
                    inputs = inp[:-1]
                    expression = []

                    try:
                        expression = expressions[inp[-1]]
                    except:
                        raise Exception("Expression not found in the dictionary!")

                    res = func(inputs, expression)
                    if(res != ([], [])):
                        expressions[exp] = res
                        self.lastEntityDict[exp] = self.lastEntitySet
                    
                if show_details:
#                     print(p, dep, inp)
                    print(res)
            
            return str(res)
        except Exception as e:
            if ignore_error:
                return None
            else:
                raise
                
    def _parse_key_value(self, key, value, typ=None):
        if typ is None:
            if(key in self.key_type.keys()):
                typ = self.key_type[key]
            else:
                typ = 'string'
        if typ=='string':
            value = ValueClass('string', value)
        elif typ=='quantity':
            if ' ' in value:
                vs = value.split()
                v = vs[0]
                unit = ' '.join(vs[1:])
            else:
                v = value
                unit = '1'
            
            value = ValueClass('quantity', float(v), unit)
        else:
            if '/' in value or ('-' in value and '-' != value[0]):
                split_char = '/' if '/' in value else '-'
                p1, p2 = value.find(split_char), value.rfind(split_char)
                y, m, d = int(value[:p1]), int(value[p1+1:p2]), int(value[p2+1:])
                value = ValueClass('date', date(y, m, d))
            else:
                value = ValueClass('year', int(value))
        return value
    
    def isId(self, entities):
        try:
            for id_ in entities:
                if(id_ is None):
                    continue
                if(id_ not in self.entities.keys()):
                    return False
            return True
        except:
            return False

    def FIND(self, inputs, expression):
        name = inputs[0]
        ## inputs[0] not in self.entity_name_to_ids then get_similar_entity_concept
        
        if(name not in self.entity_name_to_ids):
            if(name not in self.entity_name_to_ids2):
                name = nearest_neigh(name, 'e', self.idx)
                entity_ids = [self.entity_name_to_ids[name]]
            else:
                entity_ids = [self.entity_name_to_ids2[name]]
        else:         
            # print("name", name)
        # print("name170", name)
            entity_ids = [self.entity_name_to_ids[name]]
            
        if(len(entity_ids) != 0):
            self.lastEntitySet = entity_ids
        else:
            entity_ids = self.lastEntitySet    

        if(self.end == True):
            out_str = ""
            for i in range(len(entity_ids)):
                id_ = entity_ids[i]
                # name = ""
                # if(id_ in self.entities.keys()):
                #     name = self.entities[id_]['name']

                # out_str += name + "|"
                out_str += id_ + "|"
            return out_str
            
        return (entity_ids, None)
    
    def _filter_attribute(self, entity_ids, tgt_key, tgt_value, op, typ):
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")

        raw_tag_value = tgt_value
        # tgt_value = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        list_attribute = []

        for i in entity_ids:
            for attr_key, attr_val in self.entities[i]['relations'].items():
                k = attr_key
                list_attribute.append(k)
                if k==tgt_key:
                    for val in attr_val:
                        v = val[0]
                        if v == tgt_value:
                            res_ids.append(i)
                            res_facts.append(attr_key)
                            break
                    
        if(res_ids == []):
            tgt_key = nearest_kv(tgt_key, 'attributes', list_attribute, self.idx)
            # tgt_value = self._parse_key_value(tgt_key, raw_tag_value, typ)
            for i in entity_ids:
                for attr_key, attr_val in self.entities[i]['relations'].items():
                    k = attr_key
                    if k==tgt_key:
                        for val in attr_val:
                            v = val[0]
                            if v == tgt_value:
                                res_ids.append(i)
                                res_facts.append(attr_key)
                                break

        if(len(res_ids) != 0):
            self.lastEntitySet = res_ids
            self.lastFactDict = {}
            for i in range(len(res_ids)):
                if(res_ids[i] in self.lastFactDict):
                    self.lastFactDict[res_ids[i]].append(res_facts[i])
                else:
                    self.lastFactDict[res_ids[i]] = [res_facts[i]]

        else:
            res_ids = self.lastEntitySet
            res_facts = self.lastFactDict

        if(self.end == True):
            # print('here 314')
            out_str = ""
                
            for i in range(len(res_ids)):
                id_ = res_ids[i]
                # name = ""
                # if(id_ in self.entities.keys()):
                #     name = self.entities[id_]['name']
                out_str += id_ + "|"
            return out_str

        return (res_ids, res_facts)
    
    def FILTERSTR(self, inputs, expression):
        entity_ids, _ = expression

        if(len(inputs) >= 3):
            key, value, op = inputs[0], inputs[1], inputs[2]
        else:
            key, value, op = inputs[0], inputs[1], '='
        return self._filter_attribute(entity_ids, key, value, op, 'string')
    
    def RELATE(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")

        res_ids = []

        for id_ in entity_ids:
            temp_inputs = inputs
            if(id_ is None):
                continue
            else:
                out_ids = self.RELATEHELPER(temp_inputs, id_)
                if(len(out_ids[0])):
                    res_ids += out_ids[0]
                    
        if(len(res_ids) != 0):
            self.lastEntitySet = res_ids
        else:
            res_ids = self.lastEntitySet

        if(self.end == True):
            out_str = ""
            
            for i in range(len(res_ids)):
                id_ = res_ids[i]
                # name = ""
                # if(id_ in self.entities.keys()):
                #     name = self.entities[id_]['name']
                out_str += id_ + "|"
            return out_str
        
        
        return (res_ids, None)

    def RELATEHELPER(self, inputs, expression):
        entity_id = expression
        predicate = inputs[0]
        res_ids = []
        list_relation = []
        if entity_id in self.entities:
            rel_infos = self.entities[entity_id]['relations']
        
        ## if predicate not in rel_info then take nearest using llm    
        for rel_info in rel_infos.keys():
            list_relation.append(rel_info)
            if rel_info == predicate:
                value = rel_infos[rel_info]
                for v in value:
#                     res_ids.append(self.entity_name_to_ids[v[0]])
                    res_ids.append(v[0])

        if(res_ids == []):
            predicate = nearest_kv(predicate, 'relation', list_relation, self.idx)
#             predicate = inputs[0]
            res_ids = []

            if entity_id in self.entities:
                rel_infos = self.entities[entity_id]['relations']

            ## if predicate not in rel_info then take nearest using llm    
            for rel_info in rel_infos.keys():
                if rel_info == predicate:
                    value = rel_infos[rel_info]
                    for v in value:
#                         res_ids.append(self.entity_name_to_ids[v[0]])
                        res_ids.append(v[0])

        return (res_ids, None)


    def WHAT(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or len(entity_ids) == 0):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")

        name = ""
            
        for id_ in entity_ids:
            if(id_ is None):
                continue
            else:
                # name += self.entities[id_]['name'] + "|"
                name += id_ + "|"
        if(len(name)):
            name = name[:-1]
        return name


def get_program(inp):
    program = inp['program']
    func = []
    inputs = []
    dep = []
    for i in range(len(program)):
        func.append(program[i]['function'])
        inputs.append(program[i]['inputs'])
        dep.append(program[i]['dependencies'])
    return func, inputs, dep

def get_pred_program(clean_program):
    clean_prediction = []
    programs = []
    inputs = []
    for i in range(len(clean_program)):
        steps = clean_program[i]
        func = []
        inps = []
        for k in range(len(steps)):
            try:
                temp = steps[k][1]
                if(temp == "STOP"):
                    clean_program[i][k][1] = "STOP"
                    clean_program[i][k].append([''])
                    break
                flag = False
                start = 0
                out = []
                count = 0
                for j in range(len(temp)):
                    if((not flag and temp[j] == ',') or (temp[j] == ',' and temp[j-1] == "'")):
                        out.append(temp[start:j].strip())
                        start = j+1
                    elif(temp[j] == "("):
                        count += 1
                        if(count == 1):
                            out.append(temp[start:j].strip())
                            start = j+1
                    elif(temp[j] == ")"):
                        if(count == 1):
                            break
                        else:
                            count -= 1
                    elif(temp[j] == "'"):
                        flag = not flag
                out.append(temp[start:j].strip())
                clean_program[i][k][1] = out[0]
                for l in range(1, len(out)):
                    out[l] = out[l].strip("'")
                clean_program[i][k].append(out[1:])
            except:
                print("error get_pred_program at idx", i)
#                 raise
                continue
        # break
    return clean_program


def getErrorIndex(log_file):
    f = open(log_file, 'r')
    logs = f.readlines()
    f.close()
    out = []
    idx = 0
    for i in range(len(logs)):
        line = logs[i]
        if(line == '-----------------------------------\n'):
            result = logs[i-1].split()
            if(result[0] != "Matched:"):
                out.append(idx)
            idx += 1
    return out

def getErrorIndexMeta(log_file):
    f = open(log_file, 'r')
    logs = f.readlines()
    f.close()
    out = []
    idx = 0
    for i in range(len(logs)):
        line = logs[i]
        if(line == '-----------------------------------\n'):
            result = logs[i-1].split()
            if(result[0] != "ground"):
                out.append(idx)
            idx += 1
    return out

def processKB(kb):
    entity_name_to_ids = {}
    entity_name_to_ids2 = {}
    entities = {}
    mapping = pd.read_csv('../../dataset/webQSP/mid2name.tsv',sep='\t', header=None)
    mapping2 = pd.read_csv('../../dataset/webQSP/mapping_new.csv')
    
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

def clean_steps_from_new_prompt(filename):
    f = open(filename, "r")
    output = f.read()
    f.close()
    programs = output.split("Output:\n")[1:]
    clean_program = []
    for j in range(len(programs)):
        steps = []
        program = programs[j].split("Test Example:\n")
        program = program[1].split('Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:\n')
        program = program[1].split("\n")
        if(program[-1] == ''):
            program = program[:-1]
        # print(j)
        for i in range(len(program)):
            try:
                step = program[i].split('=')
                step[1] = "=".join(step[1:])
            except:
                print('Error! Invalid Expression at idx', j)
                step = ['', '']
                step[0] = "expression"
                step[1] = "STOP"
            steps.append([step[0].strip(), step[1].strip()])


        clean_program.append(steps)
        # break
    return clean_program

def clean_steps_from_new_prompt_from_json(filename):
    f = open(filename, "r")
    programs = json.load(f)
    f.close()
    clean_program = []
    for j in range(len(programs)):
        steps = []
        program = programs[j].split("\n")
        if(program[0] == "```"):
            program = program[1:]
        if(program[-1] == ''):
            program = program[:-1]
        for i in range(len(program)):
            try:
                step = program[i].split('=')
                step[1] = "=".join(step[1:])
                if("=".join(step[1:]) == "" and i == len(program)-1):
                    step[1] = "STOP"
            except:    
                # print(j)
                step = ['', '']
                step[0] = "expression"
                step[1] = "STOP"
            steps.append([step[0].strip(), step[1].strip()])
        clean_program.append(steps)
        # break
    return clean_program

def extractFinalAnswer(filename):
    f = open(filename, "r")
    val = json.load(f)
    f.close()
    answers = []
    for i in range(len(val['Questions'])):
        answer = []
        for j in range(len(val['Questions'][i]['Parses'])):
            for k in range(len(val['Questions'][i]['Parses'][j]['Answers'])):
                answer.append(val['Questions'][i]['Parses'][j]['Answers'][k]['AnswerArgument'])
        answer = list(set(answer))
        answers.append(answer)
    return answers

def main():
    matched = 0
    avg_precision = 0
    avg_recall = 0
    inp_file =  sys.argv[1] # Path of generated code file
    ground_file = sys.argv[2] # Path of ground truth file
    mode = sys.argv[3]
    
    clean_programs = clean_steps_from_new_prompt_from_json(inp_file)
    clean_programs = get_pred_program(clean_programs)

    f = open("../data/webqsp/raw_kb.txt", "r")
    kb = f.readlines()
    f.close()
    
    ground_answers = extractFinalAnswer(ground_file)
    entity_name_to_ids, entities, entity_name_to_ids2 = processKB(kb)
    
    rule_executor = RuleExecutorFunctional()
    if(mode == 'exec'):
        rule_executor = RuleExecutor(entities, entity_name_to_ids, entity_name_to_ids2)
    for i in tqdm(range(len(clean_programs))):
        idx = i
        print('idx:', idx)
        try:
            if(mode == 'val'):
                pred_func = rule_executor.forward(clean_programs[idx], idx, ignore_error=False, show_details=False)
                if(pred_func[0] == None and pred_func[1] == None):
                    print("Error! Functional signature not matched!")
                elif(rule_executor.get_passFlag()):
                    print("Passed!")
            elif(mode == 'exec'):
                pred = rule_executor.forward(clean_programs[i], idx, ignore_error=False, show_details=False)
                pred = list(set(pred.split('|')))
                
                ground_answer = ground_answers[i]
                flag = False
                
                intermediate_results = []
                for j in range(len(pred)):
                    curr_pred = pred[j]
                    if(curr_pred == ''):
                        continue

                    if(entities[curr_pred]['name'] == curr_pred):
                        relations = entities[curr_pred]['relations']
                        for k, v in relations.items():
                            for z in range(len(v)):
                                intermediate_results.append(v[z][0])
                
                pred += intermediate_results
                pred = list(set(pred))
                        
                
                for j in range(len(pred)): 
                    if pred[j] in ground_answer:
                        print("Matched:", pred)
                        matched += 1
                        flag = True
                        break
                if(not flag):
                    print("Not matched!", pred, ground_answer)
            else:
                print("Valid modes: val, exec")
        except Exception as e:
            print('Error! ', e)
#             raise
        print('-----------------------------------')
    return 0

if __name__ == '__main__':
    main()
