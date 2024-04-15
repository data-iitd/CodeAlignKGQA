from utils.value_class import ValueClass, comp, isOp
from utils.misc import invert_dict
from utils.find_closest_meta import nearest_neigh, nearest_kv, nearest_neigh_by_id
from Program.executor_abalation_functinal_new import RuleExecutorFunctional
import os
import sys
import json
from collections import defaultdict
from datetime import date
from queue import Queue
from tqdm import tqdm

class RuleExecutor(object):
    def __init__(self, entities, entity_name_to_ids):
        # print('load kb')
        self.entities = entities
        self.idx = -1
        self.end = False
        self.lastEntitySet = []

        # replace adjacent space and tab in name
        for ent_id, ent_info in self.entities.items():
            ent_info['name'] = ' '.join(ent_info['name'].split())

        self.entity_name_to_ids = entity_name_to_ids

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
#                   print(p, dep, inp)
                    print(res)

            return str(res)
        
        except Exception as e:
            if ignore_error:
                return None
            else:
                raise
    
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
            name = nearest_neigh(name, 'e', self.idx)
        entity_ids = [self.entity_name_to_ids[name]]

        if(len(entity_ids) != 0):
            self.lastEntitySet = entity_ids
        else:
            entity_ids = self.lastEntitySet

        if(self.end == True):
            out_str = ""
            for i in range(len(entity_ids)):
                id_ = entity_ids[i]
                name = ""
                if(id_ in self.entities.keys()):
                    name = self.entities[id_]['name']

                out_str += name + "|"
            return out_str
  
        return (entity_ids, None)
    
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
                name = ""
                if(id_ in self.entities.keys()):
                    name = self.entities[id_]['name']
                out_str += name + "|"
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
                    res_ids.append(self.entity_name_to_ids[v[0]])

        if(res_ids == []):
            predicate = nearest_kv(predicate, 'relation', list_relation, self.idx)
            # predicate = inputs[0]
            res_ids = []

            if entity_id in self.entities:
                rel_infos = self.entities[entity_id]['relations']

            ## if predicate not in rel_info then take nearest using llm    
            for rel_info in rel_infos.keys():
                if rel_info == predicate:
                    value = rel_infos[rel_info]
                    for v in value:
                        res_ids.append(self.entity_name_to_ids[v[0]])

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
                name += self.entities[id_]['name'] + "|"

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
    entities = {}
    id_ = 1
    for line in kb:
        line = line[:-1]
        kb_elements = line.split('|')
        if kb_elements[0] not in entity_name_to_ids.keys():
            entity_name_to_ids[kb_elements[0]] = id_
            entities[id_] = {'name': kb_elements[0], 'relations': {kb_elements[1]: [[kb_elements[2], 'forward']]}}
            id_ += 1
        else:
            enitiy_id = entity_name_to_ids[kb_elements[0]]
            if(kb_elements[1] in entities[enitiy_id]['relations']):
                entities[enitiy_id]['relations'][kb_elements[1]].append([kb_elements[2], 'forward'])
            else:
                entities[enitiy_id]['relations'][kb_elements[1]] = [[kb_elements[2], 'forward']]

        if kb_elements[2] not in entity_name_to_ids.keys():
            entity_name_to_ids[kb_elements[2]] = id_
            entities[id_] = {'name': kb_elements[2], 'relations': {kb_elements[1]: [[kb_elements[0], 'backward']]}}
            id_ += 1
        else:
            enitiy_id = entity_name_to_ids[kb_elements[2]]
            if(kb_elements[1] in entities[enitiy_id]['relations']):
                entities[enitiy_id]['relations'][kb_elements[1]].append([kb_elements[0], 'backward'])
            else:
                entities[enitiy_id]['relations'][kb_elements[1]] = [[kb_elements[0], 'backward']]
        
    return entity_name_to_ids, entities

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
    answers = f.readlines()
    f.close()
    out = []
    for ans in answers:
        final_ans = ans.split('\t')[1][:-1]
        final_ans = final_ans.split('|')
        out.append(final_ans)
        # break
    return out

def main():
    matched = 0
    avg_precision = 0
    avg_recall = 0

    inp_file = sys.argv[1] # Path of generated code file
    ground_file = sys.argv[2] # Path of ground truth file
    mode = sys.argv[3]

    clean_programs = clean_steps_from_new_prompt_from_json(inp_file)
    clean_programs = get_pred_program(clean_programs)
    
    f = open("../data/metaQA/kb.txt", "r")
    kb = f.readlines()
    f.close()
    
    ground_answers = extractFinalAnswer(ground_file)
    rule_executor = RuleExecutorFunctional()
    if(mode == 'exec'):
        entity_name_to_ids, entities = processKB(kb)
        rule_executor = RuleExecutor(entities, entity_name_to_ids)

    for i in tqdm(range(0, len(clean_programs))):
        print('idx:', i)
        try:
            if(mode == 'val'):
                pred_func = rule_executor.forward(clean_programs[i], i, ignore_error=False, show_details=False)
                if(pred_func[0] == None and pred_func[1] == None):
                    print("Error! Functional signature not matched!")
                elif(rule_executor.get_passFlag()):
                    print("Passed!")
            elif(mode == 'exec'):
                idx = i
                # program, inputs, depend = get_program(val[idx])
                pred = rule_executor.forward(clean_programs[i], idx, ignore_error=False, show_details=False)

                pred = list(set(pred.split('|')))
                ground_answer = list(set(ground_answers[i]))
                flag = False

                for j in range(len(pred)):
                    if pred[j] in ground_answer:
                        print("Matched:", pred)
                        matched += 1
                        flag = True
                        break

                if(not flag):
                    print("Not matched!", pred, ground_answer)

        except Exception as e:
            print('Error! ', e)
#             raise
        print('-----------------------------------')
    return 0

if __name__ == '__main__':
    main()
