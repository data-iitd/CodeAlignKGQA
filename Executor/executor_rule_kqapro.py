from utils.value_class import ValueClass, comp, isOp
from utils.misc import invert_dict
from utils.find_closest import nearest_neigh, nearest_kv, nearest_neigh_by_id, calculate_cosine_similarity_opt_new
from Executor.executor_abalation_functinal_new import RuleExecutorFunctional
import os
import json
from collections import defaultdict
from datetime import date
from queue import Queue
from tqdm import tqdm
import sys


class RuleExecutor(object):
    def __init__(self, kb_json):
        # self.vocab = vocab
        # print('load kb')
        kb = json.load(open(kb_json))
        self.concepts = kb['concepts']
        self.entities = kb['entities']
        self.idx = -1
        self.end = False
        self.lastEntitySet = []
        self.lastFactDict = {}
        self.lastEntityDict = {}

        # replace adjacent space and tab in name
        for con_id, con_info in self.concepts.items():
            con_info['name'] = ' '.join(con_info['name'].split())
        for ent_id, ent_info in self.entities.items():
            ent_info['name'] = ' '.join(ent_info['name'].split())

        self.entity_name_to_ids = defaultdict(list)
        for ent_id, ent_info in self.entities.items():
            self.entity_name_to_ids[ent_info['name']].append(ent_id)
        self.concept_name_to_ids = defaultdict(list)
        for con_id, con_info in self.concepts.items():
            self.concept_name_to_ids[con_info['name']].append(con_id)

        self.concept_to_entity = defaultdict(set)
        self.entity_to_concept = defaultdict(set)
        for ent_id in self.entities:
            for c in self._get_all_concepts(ent_id): # merge entity into ancestor concepts
                self.concept_to_entity[c].add(ent_id)
                self.entity_to_concept[ent_id].add(c)
        self.concept_to_entity = { k:list(v) for k,v in self.concept_to_entity.items() }
        self.entity_to_concept = { k:list(v) for k,v in self.entity_to_concept.items() }

        self.key_type = {}
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                self.key_type[attr_info['key']] = attr_info['value']['type']
                for qk in attr_info['qualifiers']:
                    for qv in attr_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
            
       
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                for qk in rel_info['qualifiers']:
                    for qv in rel_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
        # Note: key_type is one of string/quantity/date, but date means the key may have values of type year
        self.key_type = { k:v if v!='year' else 'date' for k,v in self.key_type.items() }
        self.allUnits = set([])

        # parse values into ValueClass object
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                attr_info['value'] = self._parse_value(attr_info['value'])
                self.allUnits.add(attr_info['value'].unit)

                for qk, qvs in attr_info['qualifiers'].items():
                    out_qvs = []
                    for qv in qvs:
                        pv = self._parse_value(qv)
                        out_qvs.append(pv)
                        self.allUnits.add(pv.unit)
                    attr_info['qualifiers'][qk] = out_qvs
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                for qk, qvs in rel_info['qualifiers'].items():
                    out_qvs = []
                    for qv in qvs:
                        pv = self._parse_value(qv)
                        out_qvs.append(pv)
                        self.allUnits.add(pv.unit)

                    rel_info['qualifiers'][qk] = out_qvs

        self.allUnits = list(self.allUnits)
        # some entities may have relations with concepts, we add them into self.concepts for visiting convenience
        for ent_id in self.entities:
            for rel_info in self.entities[ent_id]['relations']:
                obj_id = rel_info['object']
                if obj_id in self.concepts:
                    if 'relations' not in self.concepts[obj_id]:
                        self.concepts[obj_id]['relations'] = []
                    self.concepts[obj_id]['relations'].append({
                        'relation': rel_info['relation'],
                        'predicate': rel_info['relation'], # predicate
                        'direction': 'forward' if rel_info['direction']=='backward' else 'backward',
                        'object': ent_id,
                        'qualifiers': rel_info['qualifiers'],
                        })

    def _parse_value(self, value):
        if value['type'] == 'date':
            x = value['value']
            p1, p2 = x.find('/'), x.rfind('/')
            y, m, d = int(x[:p1]), int(x[p1+1:p2]), int(x[p2+1:])
            result = ValueClass('date', date(y, m, d))
        elif value['type'] == 'year':
            result = ValueClass('year', value['value'])
        elif value['type'] == 'string':
            result = ValueClass('string', value['value'])
        elif value['type'] == 'quantity':
            result = ValueClass('quantity', value['value'], value['unit'])
        else:
            raise Exception('unsupport value type')
        return result

    def _get_direct_concepts(self, ent_id):
        """
        return the direct concept id of given entity/concept
        """
        if ent_id in self.entities:
            return self.entities[ent_id]['instanceOf']
        elif ent_id in self.concepts:
            return self.concepts[ent_id]['subclassOf'] # instanceOf

    def _get_all_concepts(self, ent_id):
        """
        return a concept id list
        """
        ancestors = []
        q = Queue()
        for c in self._get_direct_concepts(ent_id):
            q.put(c)
        while not q.empty():
            con_id = q.get()
            ancestors.append(con_id)
            for c in self.concepts[con_id]['subclassOf']:  # instaceOf
                q.put(c)
        return ancestors

    def _swap_function(self, p):
        if(p == "FILTERSTR"):
            p = "QFILTERSTR"
        elif(p == "FILTERNUM"):
            p = "QFILTERNUM"
        elif(p == "FILTERYEAR"):
            p = "QFILTERYEAR"
        elif(p == "FILTERDATE"):
            p = "QFILTERDATE"
        elif(p == "QFILTERSTR"):
            p = "FILTERSTR"
        elif(p == "QFILTERNUM"):
            p = "FILTERNUM"
        elif(p == "QFILTERYEAR"):
            p = "FILTERYEAR"
        elif(p == "QFILTERDATE"):
            p = "FILTERDATE"
        return p
    
    def _isValidId(self, entities):
        try:
            for id_ in entities:
                if(id_ is None):
                    continue
                if(id_ not in self.entities.keys() and id_ not in self.concepts.keys()):
                    return False
            return True
        except:
            return False
        
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
            ## Execution
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
                    print(p)
                    if(p == "QUERYRELATION"):
                        inputs = inp[:-2]
                        expression_1 = []
                        expression_2 = []
                        try:
                            expression_1 = expressions[inp[-2]]
                            expression_2 = expressions[inp[-1]]
                        except:
                            raise Exception("Expression not found in the dictionary!")
                        if(len(inputs) >= 1 and inputs[0] != ''):
                            p = "RELATE"
                            print("Executing:", p)
                        
                    elif(p == "RELATE"):
                        inputs = inp[:-1]
                        expression = []
                        try:
                            expression = expressions[inp[-1]]
                        except:
                            raise Exception("Expression not found in the dictionary!")
                        if(len(inputs) == 1 and inputs[0] == ''):
                            p = "QUERYRELATION"
                            print("Executing:", p)

                    elif(p in ["QUERYATTRQUALIFIER", "QUERYATTRUNDERCONDITION"]):
                        inputs = inp[:-1]
                        expression = []
                        try:
                            expression = expressions[inp[-1]]
                        except:
                            raise Exception("Expression not found in the dictionary!")
                        if(len(inputs) >= 3 and isOp(inputs[2])):
                            p = "FILTERSTR"
                            print("Executing:", p)

                    if(p == "QUERYATTRUNDERCONDITION"):
                        inputs = inp[:-1]
                        expression = []
                        execute_again = False
                        try:
                            expression = expressions[inp[-1]]
                        except:
                            raise Exception("Expression not found in the dictionary!")
                         
                        try:
                            func = getattr(self, p)
                            res = func(inputs, expression)
                            if(res != ''):
                                expressions[exp] = res
                                self.lastEntityDict[exp] = self.lastEntitySet
                        except:
                            execute_again = True

                        if(res == ''):
                            execute_again = True

                        if(execute_again):
                            inputs = inp[:-1]
                            expression = []
                            try:
                                expression = expressions[inp[-1]]
                            except:
                                raise Exception("Expression not found in the dictionary!")
                         
                            p = "QUERYATTRQUALIFIER"
                            print("Executing:", p)
                            inputs[1], inputs[2] = inputs[2], inputs[1]
                            func = getattr(self, p)
                            res = func(inputs, expression)
                            expressions[exp] = res
                            self.lastEntityDict[exp] = self.lastEntitySet
                        
                    
                    elif(p == "QUERYATTRQUALIFIER"):
                        inputs = inp[:-1]
                        expression = []
                        try:
                            expression = expressions[inp[-1]]
                        except:
                            raise Exception("Expression not found in the dictionary!")
                        try:
                            func = getattr(self, p)
                            res = func(inputs, expression)
                            if(res != ''):
                                expressions[exp] = res
                                self.lastEntityDict[exp] = self.lastEntitySet
                        except:
                            execute_again = True

                        if(res == ''):
                            execute_again = True

                        if(execute_again):
                            inputs = inp[:-1]
                            expression = []
                            try:
                                expression = expressions[inp[-1]]
                            except:
                                raise Exception("Expression not found in the dictionary!")
                         
                            p = "QUERYATTRUNDERCONDITION"
                            print("Executing:", p)
                            inputs[1], inputs[2] = inputs[2], inputs[1]
                            func = getattr(self, p)
                            res = func(inputs, expression)
                            self.lastEntityDict[exp] = self.lastEntitySet
                            expressions[exp] = res
                    else:
                        func = getattr(self, p)
                        res = []
                        if(p in ["AND", "OR", "SELECTBETWEEN", "QUERYRELATION", "QUERYRELATIONQUALIFIER"]):
                            inputs = inp[:-2]
                            expression_1 = []
                            expression_2 = []
                            try:
                                expression_1 = expressions[inp[-2]]
                                expression_2 = expressions[inp[-1]]

                                if(len(expression_1) < 2 or (len(expression_1[0]) == 0) or not self._isValidId(expression_1[0])):
                                    expression_1 = (self.lastEntityDict[inp[-2]], None)
                                    print("Using last entity set!")
                                
                                if(len(expression_2) < 2 or (len(expression_1[0]) == 0) or not self._isValidId(expression_2[0])):
                                    expression_2 = (self.lastEntityDict[inp[-1]], None)
                                    print("Using last entity set!")



                            except:
                                raise Exception("Expression not found in the dictionary!")
                            
                            res = func(inputs, expression_1, expression_2)
                            expressions[exp] = res
                            self.lastEntityDict[exp] = self.lastEntitySet
                        else:
                            inputs = inp[:-1]
                            expression = []
                            execute_again = False
                            try:
                                expression = expressions[inp[-1]]
                            except:
                                raise Exception("Expression not found in the dictionary!")
        
                            try: 
                                res = func(inputs, expression)
                                if(res != ([], [])):
                                    expressions[exp] = res
                                    self.lastEntityDict[exp] = self.lastEntitySet
                            except Exception as e:
                                if(p in ["FILTERSTR", "FILTERNUM", "FILTERYEAR", "FILTERDATE", "QFILTERSTR", "QFILTERNUM", "QFILTERYEAR", "QFILTERDATE"]):
                                    p = self._swap_function(p)
                                    execute_again = True
                                else:
                                    raise e
                            
                            if(res == ([], []) and (p in ["FILTERSTR", "FILTERNUM", "FILTERYEAR", "FILTERDATE", "QFILTERSTR", "QFILTERNUM", "QFILTERYEAR", "QFILTERDATE"])):
                                p = self._swap_function(p)
                                execute_again = True
                            
                            if(execute_again):
                                inputs = inp[:-1]
                                expression = []
                                try:
                                    expression = expressions[inp[-1]]
                                except:
                                    raise Exception("Expression not found in the dictionary!")
                                prev_res = res
                                print("Executing: ", p)
                                func = getattr(self, p)
                                try:
                                    res = func(inputs, expression)
                                except Exception as e:
                                    if(prev_res == ([], [])):
                                        res = prev_res
                                    else:
                                        raise e
                                    
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

            if(unit not in self.allUnits):
                unit = calculate_cosine_similarity_opt_new(unit, self.allUnits)[0][1]

            try:
                value = ValueClass('quantity', float(v), unit)
            except:
                value = ValueClass('string', value)
                print('Unable to convert into quantity!')
        else:
            if '/' in value or ('-' in value and '-' != value[0]):
                split_char = '/' if '/' in value else '-'
                p1, p2 = value.find(split_char), value.rfind(split_char)
                y, m, d = int(value[:p1]), int(value[p1+1:p2]), int(value[p2+1:])
                try:
                    value = ValueClass('date', date(y, m, d))
                except:
                    value = ValueClass('string', value)
                    print('Unable to convert into date!')
            else:
                try:
                    value = ValueClass('year', int(value))
                except:
                    value = ValueClass('string', value)
                    print('Unable to convert into year!')
        return value
    
    def isId(self, entities):
        try:
            for id_ in entities:
                if(id_ is None):
                    continue
                if(id_ not in self.entities.keys() and id_ not in self.concepts.keys()):
                    return False
            return True
        except:
            return False
    
    def queryNames(self, entity_ids):
        out_str = ""
        for i in range(len(entity_ids)):
            id_ = entity_ids[i]
            name = ""
            if(id_ in self.entities.keys()):
                name = self.entities[id_]['name']
            elif(id_ in self.concepts.keys()):
                name = self.concepts[id_]['name']
            out_str += name + "|"
        if(len(out_str)):
            out_str = out_str[:-1]
        return out_str

    def FINDALL(self, inputs, expression):
        entity_ids = list(self.entities.keys())
        self.lastEntitySet = entity_ids
        return (entity_ids, None)

    def FIND(self, inputs, expression):
        name = inputs[0]
        ## inputs[0] not in self.entity_name_to_ids then get_similar_entity_concept
        if(name not in self.entity_name_to_ids and name not in self.concept_name_to_ids):
            name = nearest_neigh(name, 'e', self.idx)
        entity_ids = self.entity_name_to_ids[name]
        if name in self.concept_name_to_ids: # concept may appear in some relations
            entity_ids += self.concept_name_to_ids[name]

        if(self.end == True):
            return self.queryNames(entity_ids)

        self.lastEntitySet = entity_ids    
        return (entity_ids, None)

    def FILTERCONCEPT(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")
    
        concept_name = inputs[0]
         ## inputs[0] not in self.concept_name_to_ids then get_similar_entity_concept\
        flag = False
        if(concept_name not in self.concept_name_to_ids):
            # if(concept_name in self.entity_name_to_ids):
            #     return self.Find(dependencies, inputs)
            flag = True
            concept_name = nearest_neigh(concept_name, 'c', self.idx)

        concept_ids = self.concept_name_to_ids[concept_name]
        entity_ids_2 = []
        for i in concept_ids:
            entity_ids_2 += self.concept_to_entity[i]
        entity_ids = list(set(entity_ids) & set(entity_ids_2))
        if(entity_ids == [] and flag):
            ids = []
            entity_ids = expression
            if(not self.isId(entity_ids) or entity_ids == []):
                entity_ids = self.lastEntitySet
                print("Using last entity set!")
            concept_name = inputs[0]
            for id_ in entity_ids:
                ids += self.entity_to_concept[id_]
            ids = list(set(ids))
            concept_name = nearest_neigh_by_id(concept_name, ids, 'c', self.idx)
            concept_ids = self.concept_name_to_ids[concept_name]
            entity_ids_2 = []
            for i in concept_ids:
                entity_ids_2 += self.concept_to_entity[i]
            entity_ids = list(set(entity_ids) & set(entity_ids_2))

        if(self.end == True):
            return self.queryNames(entity_ids)
        
        if(len(entity_ids)):
            self.lastEntitySet = entity_ids      
        return (entity_ids, None)

    def _filter_attribute(self, entity_ids, tgt_key, tgt_value, op, typ):
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")

        raw_tag_value = tgt_value
        tgt_value = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        list_attribute = []

        for i in entity_ids:
            for attr_info in self.entities[i]['attributes']:
                k, v = attr_info['key'], attr_info['value']
                list_attribute.append(k)
                if k==tgt_key and v.can_compare(tgt_value) and comp(v, tgt_value, op):
                    res_ids.append(i)
                    res_facts.append(attr_info)
                    
        if(res_ids == []):
            tgt_key = nearest_kv(tgt_key, 'attributes', list_attribute, self.idx)
            tgt_value = self._parse_key_value(tgt_key, raw_tag_value, typ)
            for i in entity_ids:
                for attr_info in self.entities[i]['attributes']:
                    k, v = attr_info['key'], attr_info['value']
                    if k==tgt_key and v.can_compare(tgt_value) and comp(v, tgt_value, op):
                        res_ids.append(i)
                        res_facts.append(attr_info)

        if(len(res_ids)):
            self.lastEntitySet = res_ids
            self.lastFactDict = {}
            for i in range(len(res_ids)):
                if(res_ids[i] in self.lastFactDict):
                    self.lastFactDict[res_ids[i]].append(res_facts[i])
                else:
                    self.lastFactDict[res_ids[i]] = [res_facts[i]]

        if(self.end == True):
            return self.queryNames(self.lastEntitySet)

        return (res_ids, res_facts)

    def FILTERSTR(self, inputs, expression):
        entity_ids, _ = expression

        if(len(inputs) >= 3):
            key, value, op = inputs[0], inputs[1], inputs[2]
        else:
            key, value, op = inputs[0], inputs[1], '='
        return self._filter_attribute(entity_ids, key, value, op, 'string')

    def FILTERNUM(self, inputs, expression):
        entity_ids, _ = expression

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_attribute(entity_ids, key, value, op, 'quantity')

    def FILTERYEAR(self, inputs, expression):
        entity_ids, _ = expression

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_attribute(entity_ids, key, value, op, 'year')

    def FILTERDATE(self, inputs, expression):
        entity_ids, _ = expression

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_attribute(entity_ids, key, value, op, 'date')

    def _filter_qualifier(self, entity_ids, facts, tgt_key, tgt_value, op, typ):
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")
        
        if(facts == None or facts == []):
            facts = []
            new_entity_ids = []
            print("Using last fact set!")
            for i in range(len(entity_ids)):
                if(entity_ids[i] in self.lastFactDict):
                    for f in self.lastFactDict[entity_ids[i]]:
                        new_entity_ids.append(entity_ids[i])
                        facts.append(f)

            if(len(new_entity_ids)):
                entity_ids = new_entity_ids
            else:
                facts = None

        raw_tgt_value = tgt_value
        tgt_value = self._parse_key_value(tgt_key, tgt_value, typ)
        res_ids = []
        res_facts = []
        list_qualifier = []
         ## if tgt_key not in qualifiers, take nearest using llm
        for i, f in zip(entity_ids, facts):
            for qk, qvs in f['qualifiers'].items():
                list_qualifier.append(qk)
                if qk == tgt_key:
                    for qv in qvs:
                        if qv.can_compare(tgt_value) and comp(qv, tgt_value, op):
                            res_ids.append(i)
                            res_facts.append(f)
        if(res_ids == []):
            # print('list:', list_qualifier)
            tgt_key = nearest_kv(tgt_key, 'qualifiers', list_qualifier, self.idx)
            tgt_value = self._parse_key_value(tgt_key, raw_tgt_value, typ)
            for i, f in zip(entity_ids, facts):
                for qk, qvs in f['qualifiers'].items():
                    if qk == tgt_key:
                        for qv in qvs:
                            if qv.can_compare(tgt_value) and comp(qv, tgt_value, op):
                                res_ids.append(i)
                                res_facts.append(f)

        if(len(res_ids)):
            self.lastEntitySet = res_ids
            self.lastFactDict = {}
            for i in range(len(res_ids)):
                if(res_ids[i] in self.lastFactDict):
                    self.lastFactDict[res_ids[i]].append(res_facts[i])
                else:
                    self.lastFactDict[res_ids[i]] = [res_facts[i]]

        if(self.end == True):
            return self.queryNames(self.lastEntitySet)

        return (res_ids, res_facts)

    def QFILTERSTR(self, inputs, expression):
        entity_ids, facts = expression

        key, value, op = inputs[0], inputs[1], '='
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'string')

    def QFILTERNUM(self, inputs, expression):
        entity_ids, facts = expression

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'quantity')

    def QFILTERYEAR(self, inputs, expression):
        entity_ids, facts = expression

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'year')

    def QFILTERDATE(self, inputs, expression):
        entity_ids, facts = expression

        key, value, op = inputs[0], inputs[1], inputs[2]
        return self._filter_qualifier(entity_ids, facts, key, value, op, 'date')
    
    def RELATE(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")
        res_ids = []
        res_facts = []

        for id_ in entity_ids:
            temp_inputs = inputs
            if(id_ is None):
                continue
            else:
                out_ids, out_facts = self.RELATEHELPER(temp_inputs, id_)
                res_ids += out_ids
                res_facts += out_facts

        if(len(res_ids)):
            self.lastEntitySet = res_ids
            self.lastFactDict = {}
            for i in range(len(res_ids)):
                if(res_ids[i] in self.lastFactDict):
                    self.lastFactDict[res_ids[i]].append(res_facts[i])
                else:
                    self.lastFactDict[res_ids[i]] = [res_facts[i]]

        if(self.end == True):
            return self.queryNames(self.lastEntitySet)
        
        return (res_ids, res_facts)


    def RELATEHELPER(self, inputs, expression):
        entity_id = expression
        predicate = inputs[0]
        # Relate helper
        res_ids = []
        res_facts = []
        list_relation = []
        if entity_id in self.entities:
            rel_infos = self.entities[entity_id]['relations']
        else:
            rel_infos = self.concepts[entity_id]['relations']
        ## if predicate not in rel_info then take nearest using llm    
        for rel_info in rel_infos:
            list_relation.append(rel_info['relation'])
            if rel_info['relation']==predicate:
                res_ids.append(rel_info['object'])
                res_facts.append(rel_info)

        
        if(res_ids == []):
            inputs[0] = nearest_kv(predicate, 'relation', list_relation, self.idx)
            # print('line360', inputs[0])
            entity_id = expression
            predicate = inputs[0]
            res_ids = []
            res_facts = []
            if entity_id in self.entities:
                rel_infos = self.entities[entity_id]['relations']
            else:
                rel_infos = self.concepts[entity_id]['relations']
            ## if predicate not in rel_info then take nearest using llm    

            for rel_info in rel_infos:
                if rel_info['relation']==predicate:
                    res_ids.append(rel_info['object'])
                    res_facts.append(rel_info)

        return (res_ids, res_facts)

    def AND(self, inputs, expression_1, expression_2):
        entity_ids_1, _ = expression_1
        try:
            entity_ids_2, _ = expression_2
        except:
            entity_ids_2 = self.lastEntitySet

        if(not self.isId(entity_ids_1) or entity_ids_1 == []):
            entity_ids_1 = self.lastEntitySet
        if(not self.isId(entity_ids_2) or entity_ids_2 == []):
            entity_ids_2 = self.lastEntitySet

        output_set = list(set(entity_ids_1) & set(entity_ids_2)) 
        if(len(output_set)):
            self.lastEntitySet = output_set

        if(self.end == True):
            return self.queryNames(self.lastEntitySet)

        return (output_set, None)

    def OR(self, inputs, expression_1, expression_2):
        entity_ids_1, _ = expression_1
        try:
            entity_ids_2, _ = expression_2
        except:
            entity_ids_2 = self.lastEntitySet

        if(not self.isId(entity_ids_1) or entity_ids_1 == []):
            entity_ids_1 = self.lastEntitySet
        if(not self.isId(entity_ids_2) or entity_ids_2 == []):
            entity_ids_2 = self.lastEntitySet

        output_set = list(set(entity_ids_1) | set(entity_ids_2)) 
        if(len(output_set)):
            self.lastEntitySet = output_set

        if(self.end == True):
            return self.queryNames(self.lastEntitySet)

        return (output_set, None)

    def WHAT(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")

        entity_id = entity_ids[0]
        name = self.entities[entity_id]['name']
        return name

    def QUERYNAME(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")

        entity_id = entity_ids[0]
        # print(entity_ids)
        name = self.entities[entity_id]['name']
        return name

    def COUNT(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")

        return len(entity_ids)

    def SELECTBETWEEN(self, inputs, expression_1, expression_2):
        entity_ids_1, _ = expression_1
        try:
            entity_ids_2, _ = expression_2
        except:
            entity_ids_2 = self.lastEntitySet

        if(not self.isId(entity_ids_1) or entity_ids_1 == []):
            entity_ids_1 = self.lastEntitySet
        if(not self.isId(entity_ids_2) or entity_ids_2 == []):
            entity_ids_2 = self.lastEntitySet
        expression = (entity_ids_1 + entity_ids_2, [])
        return self.SELECTAMONG(inputs, expression)

    def SELECTAMONG(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")
            
        key, op = inputs[0], inputs[1]
        candidates = []
        list_attribute = []
        comparators = [['more', 'greater', 'larger', 'longer', 'greatest', 'largest', 'longest', 'biggest'], ['less', 'smaller', 'shorter', 'smallest', 'shortest', 'least']]
        for i in entity_ids:
            if(i not in self.entities.keys()):
                continue
            flag = False
            for attr_info in self.entities[i]['attributes']:
                list_attribute.append(attr_info['key'])
                if key == attr_info['key']:
                    flag = True
                    v = attr_info['value']
            if(flag):  
                candidates.append((i, v))
        if(len(candidates) == 0):
            key = nearest_kv(key, 'attributes', list_attribute, self.idx)
            for i in entity_ids:
                if(i not in self.entities.keys()):
                    continue
                flag = False
                for attr_info in self.entities[i]['attributes']:
                    if key == attr_info['key']:
                        flag = True
                        v = attr_info['value']
                if(flag):  
                    candidates.append((i, v))
        
        try:
            sort = sorted(candidates, key=lambda x: x[1])
        except:
            candidates_new = []
            for i in range(len(candidates)):
                candidates_new.append((candidates[i][0], candidates[i][1].value))
            sort = sorted(candidates_new, key=lambda x: x[1])
        
        i = sort[0][0] if (op in comparators[1]) else sort[-1][0]
        name = self.entities[i]['name']
        return name

    def QUERYATTR(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")

        # entity_id = entity_ids[0]
        list_attribute = []
        # print(entity_id, key)
        values = []

        for entity_id in entity_ids:
            flag = False
            key = inputs[0]
            for attr_info in self.entities[entity_id]['attributes']:
                list_attribute.append(attr_info['key'])
                if key == attr_info['key']:
                    v = attr_info['value']
                    values.append(v)
                    flag = True
                    break

            if(not flag):
                key = nearest_kv(key, 'attributes', list_attribute, self.idx)
                for attr_info in self.entities[entity_id]['attributes']:
                    if key == attr_info['key']:
                        v = attr_info['value']
                        values.append(v)
                        flag = True
                        break
        
        if(self.end):
            for i in range(len(values)):
                values[i] = str(values[i])
            values = "|".join(values)
            
        return values
    
    def QUERYATTRUNDERCONDITION(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")

        # entity_id = entity_ids[0]
        list_attribute = []
        list_qualifier = []
        values = []

        for entity_id in entity_ids:     
            flag = False       
            key, qual_key, qual_value = inputs[0], inputs[1], inputs[2]
            qual_value = self._parse_key_value(qual_key, qual_value)
            for attr_info in self.entities[entity_id]['attributes']:
                list_attribute.append(attr_info['key'])
                if key == attr_info['key']:
                    flag = False
                    for qk, qvs in attr_info['qualifiers'].items():
                        if qk == qual_key:
                            for qv in qvs:
                                if qv.can_compare(qual_value) and comp(qv, qual_value, "="):
                                    flag = True
                                    break
                        if flag:
                            break
                    if flag:
                        v = attr_info['value']
                        values.append(v)
                        break

            if(not flag):
                key = nearest_kv(key, 'attributes', list_attribute, self.idx)
        
                for attr_info in self.entities[entity_id]['attributes']:
                    if key == attr_info['key']:
                        for qk, qvs in attr_info['qualifiers'].items():
                            list_qualifier.append(qk)
                
                qual_key = nearest_kv(qual_key, 'qualifiers', list_qualifier, self.idx)
                qual_value = self._parse_key_value(qual_key, inputs[2])

                for attr_info in self.entities[entity_id]['attributes']:
                    if key == attr_info['key']:
                        flag = False
                        for qk, qvs in attr_info['qualifiers'].items():
                            if qk == qual_key:
                                for qv in qvs:
                                    if qv.can_compare(qual_value) and comp(qv, qual_value, "="):
                                        flag = True
                                        break
                            if flag:
                                break
                        if flag:
                            v = attr_info['value']
                            values.append(v)
                            break

            else:
                break
                    
        if(self.end):
            for i in range(len(values)):
                values[i] = str(values[i])
            values = "|".join(values)

        return values

    def _verify(self, expression, value, op, typ):
        attr_value = expression
        answer = 'no'
        value = self._parse_key_value(None, value, typ)
        if(isinstance(attr_value, list)):
            for attr_val in attr_value:
                if attr_val.can_compare(value) and comp(attr_val, value, op):
                    if(not self.end):
                        return (self.lastEntitySet, None)
                    answer = 'yes'
        else:
            if attr_value.can_compare(value) and comp(attr_value, value, op):
                if(not self.end):
                    return (self.lastEntitySet, None)
                answer = 'yes'

        if(answer == 'yes'):
            entityNames = self.queryNames(self.lastEntitySet)
            answer = answer + '|' + entityNames
        
        return answer

    def VERIFYSTR(self, inputs, expression):
        value, op = inputs[0], '='
        return self._verify(expression, value, op, 'string')
        

    def VERIFYNUM(self, inputs, expression):
        value, op = inputs[0], inputs[1]
        return self._verify(expression, value, op, 'quantity')

    def VERIFYYEAR(self, inputs, expression):
        value, op = inputs[0], inputs[1]
        return self._verify(expression, value, op, 'year')

    def VERIFYDATE(self, inputs, expression):
        value, op = inputs[0], inputs[1]
        return self._verify(expression, value, op, 'date')

    def QUERYRELATION(self, inputs, expression_1, expression_2):
        entity_ids_1, _ = expression_1
        try:
            entity_ids_2, _ = expression_2
        except:
            entity_ids_2 = self.lastEntitySet

        if(not self.isId(entity_ids_1) or entity_ids_1 == []):
            entity_ids_1 = self.lastEntitySet
        if(not self.isId(entity_ids_2) or entity_ids_2 == []):
            entity_ids_2 = self.lastEntitySet

        entity_id_1 = entity_ids_1[0]
        entity_id_2 = entity_ids_2[0]
        if entity_id_1 in self.entities:
            rel_infos = self.entities[entity_id_1]['relations']
        else:
            rel_infos = self.concepts[entity_id_1]['relations']
        p = None
        for rel_info in rel_infos:
            if rel_info['object']==entity_id_2:
                p = rel_info['relation']
        return p

    def QUERYATTRQUALIFIER(self, inputs, expression):
        entity_ids, _ = expression
        if(not self.isId(entity_ids) or entity_ids == []):
            entity_ids = self.lastEntitySet
            print("Using last entity set!")
        
        # entity_id = entity_ids[0]
        list_attribute = []
        list_qualifier = []
        q_values = []

        for entity_id in entity_ids:
            key, value, qual_key = inputs[0], inputs[1], inputs[2]
            value = self._parse_key_value(key, value)
            for attr_info in self.entities[entity_id]['attributes']:
                list_attribute.append(attr_info['key'])
                if attr_info['key']==key and attr_info['value'].can_compare(value) and \
                    comp(attr_info['value'], value, '='):
                    for qk, qvs in attr_info['qualifiers'].items():
                        if qk == qual_key:
                            q_values.append(qvs[0])
                            # return qvs[0]
            
            ## if key not in attributes then take nearest using llm  
            
            key = nearest_kv(key, 'attributes', list_attribute, self.idx)
            value = self._parse_key_value(key, inputs[1])
            
            for attr_info in self.entities[entity_id]['attributes']:
                if key == attr_info['key']:
                    for qk, qvs in attr_info['qualifiers'].items():
                        list_qualifier.append(qk)
            
            qual_key = nearest_kv(qual_key, 'qualifiers', list_qualifier, self.idx)
                    
            for attr_info in self.entities[entity_id]['attributes']:
                list_attribute.append(attr_info['key'])
                if attr_info['key']==key and attr_info['value'].can_compare(value) and \
                    comp(attr_info['value'], value, '='):
                    for qk, qvs in attr_info['qualifiers'].items():
                        if qk == qual_key:
                            q_values.append(qvs[0])
                            # return qvs[0]
        if(len(q_values) == 0):
            return None
        
        if(self.end):
            for i in range(len(q_values)):
                q_values[i] = str(q_values[i])
            q_values = "|".join(q_values)

        return q_values

    def QUERYRELATIONQUALIFIER(self, inputs, expression_1, expression_2):
        entity_ids_1, _ = expression_1
        try:
            entity_ids_2, _ = expression_2
        except:
            entity_ids_2 = self.lastEntitySet

        if(not self.isId(entity_ids_1) or entity_ids_1 == []):
            entity_ids_1 = self.lastEntitySet
        if(not self.isId(entity_ids_2) or entity_ids_2 == []):
            entity_ids_2 = self.lastEntitySet

        entity_id_1 = entity_ids_1[0]
        entity_id_2 = entity_ids_2[0]
        predicate, qual_key = inputs[0], inputs[1]

        if entity_id_1 in self.entities:
            rel_infos = self.entities[entity_id_1]['relations']
        else:
            rel_infos = self.concepts[entity_id_1]['relations']

        list_relation = []
        list_qualifier = []

        for rel_info in rel_infos:
            if rel_info['object']==entity_id_2:
                list_relation.append(rel_info['relation'])
            if rel_info['object']==entity_id_2 and \
                rel_info['relation']==predicate:
                for qk, qvs in rel_info['qualifiers'].items():
                    if qk == qual_key:
                        return qvs[0]

        ## Nearest
        predicate = nearest_kv(predicate, 'relation', list_relation, self.idx)
        
        for rel_info in rel_infos:
            if rel_info['object']==entity_id_2 and \
                rel_info['relation']==predicate:
                for qk, qvs in rel_info['qualifiers'].items():
                    list_qualifier.append(qk)
        
        qual_key = nearest_kv(qual_key, 'qualifiers', list_qualifier, self.idx)

        for rel_info in rel_infos:
            if rel_info['object']==entity_id_2:
                list_relation.append(rel_info['relation'])
            if rel_info['object']==entity_id_2 and \
                rel_info['relation']==predicate:
                for qk, qvs in rel_info['qualifiers'].items():
                    if qk == qual_key:
                        return qvs[0]

        return None


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


def main():
    val = json.load(open(os.path.join('Path of val file'))) 
    input_dir = './preprocessed_kb/'
    matched = 0
    mode = sys.argv[1]

    filename = "Path of generated code file"
    clean_programs = clean_steps_from_new_prompt_from_json(filename)
    clean_programs = get_pred_program(clean_programs)
    rule_executor = RuleExecutorFunctional()
    if(mode == 'exec'):
        rule_executor = RuleExecutor(os.path.join(input_dir, 'kb.json'))


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
                if(pred != val[idx]['answer']):
                    pred1 = pred.split('|')
                    flag = False
                    for p in pred1:
                        if(p == val[idx]['answer']):
                            print("Matched:", pred, val[idx]['answer'])
                            flag = True
                            matched += 1
                            break
                    if(not flag):
                        print("Not matched!", pred, val[idx]['answer'])
                else:
                    print("Matched:", pred)
                    matched += 1
            else:
                print("Valid modes: val, exec")
        except Exception as e:
            print('Error! ', e)
        print('-----------------------------------')

    print('Accuracy: ', matched/len(val))
    return 0

if __name__ == '__main__':
    main()
