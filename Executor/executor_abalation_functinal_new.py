from utils.value_class import ValueClass, comp, isOp
from utils.misc import invert_dict
import os
import json
from collections import defaultdict
from datetime import date
from queue import Queue
from tqdm import tqdm

class RuleExecutorFunctional(object):
    def __init__(self):
        print('init')

    def forward(self, clean_program, idx, 
                ignore_error=False, show_details=False):
        self.idx = idx
        self.end = False
        self.lastEntitySet = []
        expressions = {}
        res = ""
        prev_step = ""
        self.passFlag = True
        try:
            count_step = 0        
            for exp, p, inp in clean_program:
                if (count_step == len(clean_program)-2):
                    # print('i ', i)
                    self.end = True
                if (count_step == len(clean_program) - 1 and p != "STOP"):
                    print("Error! Last step is not STOP")
                
                count_step += 1
                if((p == "FIND" or p == "FINDALL") and prev_step != "START"):
                    print("Error! Expression should start with START!")
                    self.passFlag = False
                if p == 'START':
                    expressions[exp] = []
                    prev_step = p
                    continue
                elif p == 'STOP':
                    expressions[exp] = res
                    break
                else:
                    print(p)
                    func = getattr(self, p)
                    if(p in ["FIND", "FILTERCONCEPT", "RELATE", "QUERYATTR", "QUERYRELATION", "VERIFYSTR", "AND", "OR"] and len(inp) < 2):
                        print("Error! Insufficient paramters!")
                        self.passFlag = False
                    elif(p in ["FILTERSTR", "QFILTERSTR", "SELECTAMONG", "VERIFYNUM", "VERIFYYEAR", "VERIFYDATE", ] and len(inp) < 3):
                        print("Error! Insufficient paramters!")
                        self.passFlag = False
                    elif(p in ["FILTERNUM", "FILTERYEAR", "FILTERDATE", "QFILTERNUM", "QFILTERYEAR", "QFILTERDATE", "QUERYATTRQUALIFIER", "QUERYRELATIONQUALIFIER", "SELECTBETWEEN", "QUERYATTRUNDERCONDITION"] and len(inp) < 4):
                        print("Error! Insufficient paramters!")
                        self.passFlag = False
                    elif(p in ["COUNT", "FINDALL"] and len(inp) < 1):
                        print("Error! Insufficient paramters!")
                        self.passFlag = False

                    if(p in ["AND", "OR", "SELECTBETWEEN", "QUERYRELATION", "QUERYRELATIONQUALIFIER"]):
                        inputs = inp[:-2]
                        expression_1 = []
                        expression_2 = []
                        try:
                            expression_1 = expressions[inp[-2]]
                            expression_2 = expressions[inp[-1]]
                            if(inp[-2] == inp[-1]):
                                print("Error! Expressions should be distinct!")
                                self.passFlag = False
                        except:
                            print("Error! Expression not found in the dictionary!") 
                            self.passFlag = False
                        res = func(inputs, expression_1, expression_2)
                    else:
                        inputs = inp[:-1]
                        expression = []
                        try:
                            expression = expressions[inp[-1]]
                        except:
                            print("Error! Expression not found in the dictionary!") 
                            self.passFlag = False
                        res = func(inputs, expression)
                    if(res[0] == None and res[1] == None):
                        return res
                    expressions[exp] = res
                if show_details:
                    print(res)
                prev_step = p
            return res
        except Exception as e:
            if ignore_error:
                return None
            else:
                raise

    def get_passFlag(self):
        return self.passFlag
    
    def FINDALL(self, inputs, expression):
        return ("entity_ids", None)

    def FIND(self, inputs, expression): 
        return ("entity_ids", None)

    def FILTERCONCEPT(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return (entity_ids, None)
            else:
                return (None, None)
        except:
            return (None, None)
            

    def FILTERSTR(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return (entity_ids, "facts")
            else:
                return (None, None)
        except:
            return (None,None)

    def FILTERNUM(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return (entity_ids, "facts")
            else:
                return (None, None)
        except:
            return (None,None)

    def FILTERYEAR(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return (entity_ids, "facts")
            else:
                return (None, None)
        except:
            return (None,None)

    def FILTERDATE(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return (entity_ids, "facts")
            else:
                return (None, None)
        except:
            return (None,None)

    def QFILTERSTR(self, inputs, expression):
        try:
            entity_ids, facts = expression
            if(entity_ids == "entity_ids" and facts == "facts"):
                return (entity_ids, "facts")
            else:
                return (None, None)
        except:
            return (None,None)

    def QFILTERNUM(self, inputs, expression):
        try:
            entity_ids, facts = expression
            if(entity_ids == "entity_ids" and facts == "facts"):
                return (entity_ids, "facts")
            else:
                return (None, None)
        except:
            return (None,None)
        
    def QFILTERYEAR(self, inputs, expression):
        try:
            entity_ids, facts = expression
            if(entity_ids == "entity_ids" and facts == "facts"):
                return (entity_ids, "facts")
            else:
                return (None, None)
        except:
            return (None,None)

    def QFILTERDATE(self, inputs, expression):
        try:
            entity_ids, facts = expression
            if(entity_ids == "entity_ids" and facts == "facts"):
                return (entity_ids, "facts")
            else:
                return (None, None)
        except:
            return (None,None)

    def RELATE(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return (entity_ids, "facts")
            else:
                return (None, None)
        except:
            return (None,None)


    def AND(self, inputs, expression_1, expression_2):
        try:
            entity_ids_1, _ = expression_1
            entity_ids_2, _ = expression_2
            if(entity_ids_1 == "entity_ids" and entity_ids_2 == "entity_ids"):
                return (entity_ids_1, None)
            else:
                return (None, None)
        except:
            return (None,None)

    def OR(self, inputs, expression_1, expression_2):
        try:
            entity_ids_1, _ = expression_1
            entity_ids_2, _ = expression_2
            if(entity_ids_1 == "entity_ids" and entity_ids_2 == "entity_ids"):
                return (entity_ids_1, None)
            else:
                return (None, None)
        except:
            return (None,None)

    def WHAT(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return ("string", None)
            else:
                return (None, None)
        except:
            return (None, None)

    def QUERYNAME(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return ("string", None)
            else:
                return (None, None)
        except:
            return (None, None)

    def COUNT(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return ('length', None)
            else:
                return (None, None)
        except:
            return (None, None)

    def SELECTBETWEEN(self, inputs, expression_1, expression_2):
        try:
            entity_ids_1, _ = expression_1
            entity_ids_2, _ = expression_2
            if(entity_ids_1 == "entity_ids" and entity_ids_2 == "entity_ids"):
                return ("string", None)
            else:
                return (None, None)
        except:
            return (None,None)

    def SELECTAMONG(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return ("string", None)
            else:
                return (None, None)
        except:
            return (None,None)

    def QUERYATTR(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return ("string", None)
            else:
                return (None, None)
        except:
            return (None,None)
    
    def QUERYATTRUNDERCONDITION(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return ("string", None)
            else:
                return (None, None)
        except:
            return (None,None)

    def _verify(self, expression, value, op, typ):
        try:
            attr_value, _ = expression
            # print(attr_value)
            if(attr_value == "string"):
                if(self.end):
                    return ("bool", None)
                return ("entity_ids", None)
            else:
                return (None, None)
        except:
            return (None, None)

    def VERIFYSTR(self, inputs, expression):
        value, op = "value", '='
        return self._verify(expression, value, op, 'string')
        

    def VERIFYNUM(self, inputs, expression):
        value, op = "value", '='
        return self._verify(expression, value, op, 'quantity')

    def VERIFYYEAR(self, inputs, expression):
        value, op = "value", '='
        return self._verify(expression, value, op, 'year')

    def VERIFYDATE(self, inputs, expression):
        value, op = "value", '='
        return self._verify(expression, value, op, 'date')

    def QUERYRELATION(self, inputs, expression_1, expression_2):
        try:
            entity_ids_1, _ = expression_1
            entity_ids_2, _ = expression_2
            if(entity_ids_1 == "entity_ids" and entity_ids_2 == "entity_ids"):
                return ("string", None)
            else:
                return (None, None)
        except:
            return (None,None)

    def QUERYATTRQUALIFIER(self, inputs, expression):
        try:
            entity_ids, _ = expression
            if(entity_ids == "entity_ids"):
                return ("string", None)
            else:
                return (None, None)
        except:
            return (None,None)

    def QUERYRELATIONQUALIFIER(self, inputs, expression_1, expression_2):
        try:
            entity_ids_1, _ = expression_1
            entity_ids_2, _ = expression_2
            if(entity_ids_1 == "entity_ids" and entity_ids_2 == "entity_ids"):
                return ("string", None)
            else:
                return (None, None)
        except:
            return (None,None)


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


def getErrorIndexNew(log_file, log_file_prev):
    errorIdx = getErrorIndex(log_file_prev)
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
                out.append(errorIdx[idx])
            idx += 1
    return out
