import json
from tqdm import tqdm
import time
import requests
from get_samples_from_pool_kqa import *
import warnings
from kopl_to_program import *
# from itertools import groupby
warnings.filterwarnings("ignore")

api_key = 'ADD API KEY'

def generate_output(inp_text, model_name):
    json_payload={
                "contents": [{
                    "parts":[
                        {"text": inp_text}]}],
                "generationConfig": {
            "stopSequences": [
                "STOP"
            ],
            "temperature": 0,
            "maxOutputTokens": 800
        },
                "safetySettings": [ 
                { 
                    "category": "HARM_CATEGORY_HARASSMENT", 
                    "threshold": "BLOCK_NONE" 
                },
                { 
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE" 
                },
                { 
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                    "threshold": "BLOCK_NONE" 
                },
                { 
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT", 
                    "threshold": "BLOCK_NONE" 
                },
            ]
                }
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}'
    headers = {"Content-Type": "application/json"
                   }
    response = requests.post(url, headers=headers, json=json_payload, verify=False)
    res = response.json()
    return res

init_prompt = '''
```
You are a helpful and faithful python code generator that always follows the below specified rules: 
    - Please use the functions defined below to generate the expression corresponding to the question step by step. 
    - Use the training examples to understand the step generation process and stick only to the output format provided in the training examples. Do not generate any explanation text.
    - Do not use entities and concepts outside of the list provided in each test question. If None is mentioned in concept in question then it means that their is no concept present in the test question and you can't generate any concept related function.
    - Use Verify Functions as the last step of the program before STOP function. 
    - The datatypes are as follows:
        - entities: list of entity type
        - value: value of an attribute
        - qvalue: value of a qualifier
        - boolean: True or False
        - relation: relation name
        - facts: knowledge graph fact of the form (entity, predicate, object)
```

def START():
    ```
    Initialize the expression
    Parameters: None
    Returns: 
        expression (any): initialize expression
    ```
    return 'START()'

def FIND(entity: str, expression: any) -> entities:
    ```
    Return all entities having the input entity as name in the knowledge graph
    Parameters:
        entity (str): input entity name
        expression (any): the expression on which it will be executed
    Returns:
        expression (entities): evaluated expression of type entities
    ```
    assert isinstance(expression, any) == True
    return 'FIND({}, {})'.format(entity, expression)

def FINDALL(expression: any) -> entities:
    ```
    Return all entities in the knowledge graph
    Parameters:
        expression (any): the expression on which it will be executed
    Returns:
        expression (entities): evaluated expression of type entities
    ```
    assert isinstance(expression, any) == True
    return 'FINDALL({})'.format(expression)

def FILTERCONCEPT(concept: str, expression: entities) -> entities:
    ```
    Return entities that belongs to the input concept in the knowledge graph
    Parameters:
        concept (str): input concept name
        expression (entities): functional input from the expression
    Returns:
        expression (entities): evaluated expression of type entities
    ```
    assert isinstance(expression, entities) == True
    return 'FILTERCONCEPT({}, {})'.format(concept, expression)

def FILTERSTR(attribute: str, value: str, expression: entities) -> tuple[entities, facts]:
    ```
    Return entities with the input attribute and value of string type in the knowledge graph
    Parameters:
        attribute (str): input attribute name
        value (str): input attribute value of type string
        expression (entities): functional input from the expression of type entities
    Returns:
        expression (tuple[entities, facts]): evaluated expression of type tuple[entities, facts]
    ```
    assert isinstance(expression, entities) == True
    return 'FILTERSTR({}, {}, {})'.format(attribute, value, expression)

def FILTERNUM(attribute: str, value: int, op: str, expression: entities) -> tuple[entities, facts]:
    ```
    Return entities with the input attribute and value of integer type and op in the knowledge graph
    Parameters:
        attribute (str): input attribute name
        value (int): input attribute value of type integer
        op (str): operator to be applied
        expression (entities): functional input from the expression of type entities
    Returns:
        expression (tuple[entities, facts]): evaluated expression of type tuple[entities, facts]
    ```
    assert isinstance(expression, entities) == True
    return 'FILTERNUM({}, {}, {}, {})'.format(attribute, value, op, expression)

def FILTERYEAR(attribute: str, value: year, op: str, expression: entities) -> tuple[entities, facts]:
    ```
    Return entities with the input attribute and value of year and op in the knowledge graph
    Parameters:
        attribute (str): input attribute name
        value (year): input attribute value of type year
        op (str): operator to be applied
        expression (entities): functional input from the expression of type entities
    Returns:
        expression (tuple[entities, facts]): evaluated expression of type tuple[entities, facts]
    ```
    assert isinstance(expression, entities) == True
    return 'FILTERYEAR({}, {}, {}, {})'.format(attribute, value, op, expression)

def FILTERDATE(attribute: str, value: date, op: str, expression: entities) -> tuple[entities, facts]:
    ```
    Return entities with the input attribute and value of date and op in the knowledge graph
    Parameters:
        attribute (str): input attribute name
        value (date): input attribute value of type date
        op (str): operator to be applied
        expression (entities): functional input from the expression of type entities
    Returns:
        expression (tuple[entities, facts]): evaluated expression of type tuple[entities, facts]
    ```
    assert isinstance(expression, entities) == True
    return 'FILTERDATE({}, {}, {}, {})'.format(attribute, value, op, expression)

def QFILTERSTR(qualifier: str, qvalue: str, expression: tuple[entities, facts]) -> tuple[entities, facts]:
    ```
    Return entities with the input qualifier and qualifier value of string type in the knowledge graph
    Parameters:
        qualifier (str): input qualifier name
        qvalue (str): input qualifier value of type string
        expression (tuple[entities, facts]): functional input from the expression of type tuple[entities, facts]
    Returns:
        expression (tuple[entities, facts]): evaluated expression of type tuple[entities, facts]
    ```
    assert isinstance(expression, tuple[entities, facts]) == True
    return 'QFILTERSTR({}, {}, {})'.format(qualifier, qvalue, expression)

def QFILTERNUM(qualifier: str, qvalue: int, op: str, expression: tuple[entities, facts]) -> tuple[entities, facts]:
    ```
    Return entities with the input qualifier and qualifier value of integer type and op in the knowledge graph
    Parameters:
        qualifier (str): input qualifier name
        qvalue (int): input qualifier value of type integer
        op (str): operator to be applied
        expression (tuple[entities, facts]): functional input from the expression of type tuple[entities, facts]
    Returns:
        expression (tuple[entities, facts]): evaluated expression of type tuple[entities, facts]
    ```
    assert isinstance(expression, tuple[entities, facts]) == True
    return 'QFILTERNUM({}, {}, {}, {})'.format(qualifier, qvalue, op, expression)

def QFILTERYEAR(qualifier: str, qvalue: year, op: str, expression: tuple[entities, facts]) -> tuple[entities, facts]:
    ```
    Return entities with the input qualifier and qualifier value of year and op in the knowledge graph
    Parameters:
        qualifier (str): input qualifier name
        qvalue (int): input qualifier value of type year
        op (str): operator to be applied
        expression (tuple[entities, facts]): functional input from the expression of type tuple[entities, facts]
    Returns:
        expression (tuple[entities, facts]): evaluated expression of type tuple[entities, facts]
    ```
    assert isinstance(expression, tuple[entities, facts]) == True
    return 'QFILTERYEAR({}, {}, {}, {})'.format(qualifier, qvalue, op, expression), entities

def QFILTERDATE(qualifier: str, qvalue: date, op: str, expression: tuple[entities, facts]) -> tuple[entities, facts]:
    ```
    Return entities with the input qualifier and qualifier value of date and op in the knowledge graph
    Parameters:
        qualifier (str): input qualifier name
        qvalue (int): input qualifier value of date
        op (str): operator to be applied
        expression (tuple[entities, facts]): functional input from the expression of type tuple[entities, facts]
    Returns:
        expression (tuple[entities, facts]): evaluated expression of type tuple[entities, facts]
    ```
    assert isinstance(expression, tuple[entities, facts]) == True
    return 'QFILTERDATE({}, {}, {}, {})'.format(qualifier, qvalue, op, expression)

def RELATE(relation: str, expression: entities) -> tuple[entities, facts]:
    ```
    Return entities that have the input relation with the given entity in the knowledge graph
    Parameters:
        relation (str): input relation name
        expression (entities): functional input from the expression of type entities
    Returns:
        expression (tuple[entities, facts]): evaluated expression of type tuple[entities, facts]
    ```
    assert isinstance(expression, entities) == True
    return 'RELATE({}, {})'.format(relation, expression)

def QUERYATTR(attribute: str, expression: entities) -> value:
    ```
    Return the attribute value of the entity in the knowledge graph
    Parameters:
        attribute (str): input attribute name
        expression (entities): functional input from the expression of type entities
    Returns:
        expression (value): evaluated expression of type value
    ```
    assert isinstance(expression, entities) == True
    return 'QUERYATTR({}, {})'.format(attribute, expression)

def QUERYATTRQUALIFIER(attribute: str, value: str, key: str, expression: entities) -> qvalue:
    ```
    Return the qualifier value of the fact (Entity, Key, Value) in the knowledge graph
    Parameters:
        attribute (str): input attribute name
        value (str): input value of type string
        key (str): input key of type string
        expression (entities): functional input from the expression of type entities
    Returns:
        expression (qvalue): evaluated expression of type qvalue
    ```
    assert isinstance(expression, entities) == True
    return 'QUERYATTRQUALIFIER({}, {}, {}, {})'.format(attribute, value, key, expression)

def QUERYRELATION(expression_1: entities, expression_2: entities) -> relation:
    ```
    Return the relation between two entities in the knowledge graph
    Parameters:
        expression_1 (entities): functional input from the expression of type entities
        expression_2 (entities): functional input from another expression of type entities. expression_1 and expression_2 should be different.
    Returns:
        expression (relation): evaluated expression of type relation
    ```
    assert isinstance(expression_1, entities) == True
    assert isinstance(expression_2, entities) == True
    return 'QUERYRELATION({}, {})'.format(expression_1, expression_2)

def QUERYRELATIONQUALIFIER(relation: str, qualifier: str, expression_1: entities, expression_2: entities) -> qvalue:
    ```
    Return the qualifier value of the fact in expressions from the knowledge graph
    Parameters:
        relation (str): input relation name
        qualifier (str): input qualifier name
        expression_1 (entities): functional input from the expression of type entities
        expression_2 (entities): functional input from another expression of type entities. expression_1 and expression_2 should be different.
    Returns:
        expression (qvalue): evaluated expression of type qvalue
    ```
    assert isinstance(expression_1, entities) == True
    assert isinstance(expression_2, entities) == True
    return 'QUERYRELATIONQUALIFIER({}, {}, {}, {})'.format(relation, qualifier, expression_1, expression_2)

def SELECTBETWEEN(attribute: str, op: str, expression_1: entities, expression_2: entities) -> str:
    ```
    From the two entities, find the one whose attribute value is greater or less and return its name in the knowledge graph
    Parameters:
        attribute (str): input attribute name
        op (str): operator to be applied
        expression_1 (entities): functional input from the expression of type entities
        expression_2 (entities): functional input from another expression of type entities. expression_1 and expression_2 should be different.
    Returns:
        expression (str): evaluated expression of type string
    ```
    assert isinstance(expression_1, entities) == True
    assert isinstance(expression_2, entities) == True
    return 'SELECTBETWEEN({}, {}, {}, {})'.format(attribute, op, expression_1, expression_2)

def SELECTAMONG(attribute: str, op: str, expression: entities) -> str:
    ```
    From the entity set, find the one whose attribute value is the largest or smallest in the knowledge graph
    Parameters:
        attribute (str): input attribute name
        op (str): operator to be applied
        expression (entities): functional input from the expression of type entities
    Returns:
        expression (str): evaluated expression of type string
    ```
    assert isinstance(expression, entities) == True
    return 'SELECTAMONG({}, {}, {}, {})'.format(attribute, op, expression)

def QUERYATTRUNDERCONDITION(attribute: str, qualifier: str, value: str, expression: entities) -> value:
    ```
    Return the attribute value whose corresponding fact should satisfy the qualifier key in the knowledge graph
    Parameters:
        attribute (str): input attribute name
        qualifier (str): input qualifier name
        value (str): input value of type string
        expression (entities): functional input from the expression of type entities
    Returns:
        expression (value): evaluated expression of type value
    ```
    assert isinstance(expression, entities) == True
    return 'QUERYATTRUNDERCONDITION({}, {}, {}, {})'.format(attribute, qualifier, value, expression)

def VERIFYSTR(value: str, expression: value) -> boolean:
    ```
    Return whether the value is equal as string with the expression
    Parameters:
        value (str): input value of type string
        expression (value): functional input from the expression of type value
    Returns:
        expression (boolean): evaluated expression of type boolean
    ```
    assert isinstance(expression, value) == True
    return 'VERIFYSTR({}, {})'.format(value, expression)

def VERIFYNUM(value: int, op: str, expression: value) -> boolean:
    ```
    Return whether the value satisfies the op condtion as integer with the expression
    Parameters:
        value (str): input value of type integer
        op (str): operator to be applied
        expression (value): functional input from the expression of type value
    Returns:
        expression (boolean): evaluated expression of type boolean
    ```
    assert isinstance(expression, value) == True
    return 'VERIFYNUM({}, {}, {})'.format(value, op, expression)

def VERIFYYEAR(value: year, op: str, expression: value) -> boolean:
    ```
    Return whether the value satisfies the op condtion as year with the expression
    Parameters:
        value (str): input value of type year
        op (str): operator to be applied
        expression (value): functional input from the expression of type value
    Returns:
        expression (boolean): evaluated expression of type boolean
    ```
    assert isinstance(expression, value) == True
    return 'VERIFYYEAR({}, {}, {})'.format(value, op, expression)

def VERIFYDATE(value: date, op: str, expression: value) -> boolean:
    ```
    Return whether the value satisfy the op condition as date with the expression
    Parameters:
        value (str): input value of type integer
        op (str): operator to be applied
        expression (value): functional input from the expression of type value
    Returns:
        expression (boolean): evaluated expression of type boolean
    ```
    assert isinstance(expression, value) == True
    return 'VERIFYYEAR({}, {}, {})'.format(value, op, expression)

def AND(expression_1: entities, expression_2: entities) -> entities:
    ```
    Return the intersection of the input expressions
    Parameters:
        expression_1 (entities): functional input from the expression of type entities
        expression_2 (entities): functional input from another expression of type entities. expression_1 and expression_2 should be different.
    Returns:
        expression (entities): evaluated expression of type entities
    ```
    assert isinstance(expression_1, entities) == True
    assert isinstance(expression_2, entities) == True
    return '(AND {}, {})'.format(expression_1, expression_2)

def OR(expression_1: entities, expression_2: entities) -> entities:
    ```
    Return the union of the input expressions
    Parameters:
        expression_1 (entities): functional input from the expression of type entities
        expression_2 (entities): functional input from another expression of type entities. expression_1 and expression_2 should be different.
    Returns:
        expression (entities): evaluated expression of type entities
    ```
    assert isinstance(expression_1, entities) == True
    assert isinstance(expression_2, entities) == True
    return '(OR {}, {})'.format(expression_1, expression_2)

def COUNT(expression: entities) -> int:
    ```
    Return the count of elements
    Parameters:
        expression (entities): functional input from the expression of type entities
    Returns:
        expression (int): evaluated expression of type integer
    ```
    assert isinstance(expression, entities) == True
    return '(COUNT {})'.format(expression)

def STOP(expression: any): 
    ```
    Stop and return the expression
    ```
    return expression

Training Examples:

'''

def get_entities_concepts(program):
    entities = []
    concepts = []
    for prog in program:
        if prog['function'] == "Find":
            entities.extend(prog['inputs'])
        elif prog['function'] == "FilterConcept":
            concepts.extend(prog['inputs'])
    if len(entities) > 0 and len(concepts) > 0:
        return entities, concepts
    elif len(entities) > 0 and len(concepts) == 0:
        return entities, None
    elif len(entities) == 0 and len(concepts) > 0:
        return None, concepts
    elif len(entities) == 0 and len(concepts) == 0:
        return None, None

def get_facts(idx):
    facts_str = "[\n"
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

f = open("../data/KQAPro.IID/val.json")
data = json.load(f)
f.close()

f = open("../data/KQAPro.IID/train.json")
train_data = json.load(f)
f.close()

f1 = open("Add path of facts file retrieved using Question-specific subgraph information")
facts = f1.readlines()
f1.close()

def run(data, init_prompt):
    model_name = "gemini-pro"
    gen_steps = []
    previous_output = ""

    for i in tqdm(range(0, len(data))):
        output = ""
        entities, concepts = get_entities_concepts(data[i]['program'])
        program_steps = ""
        program = data[i]['program']
        for j in range(len(program)):
            program_steps += f"Step {j+1}: {program[j]['function']}({', '.join(program[j]['inputs'])}) | "
        target_question = program_steps
        target_embeddings = model.encode(target_question)
        results = collection.query(
            query_embeddings=target_embeddings.tolist(),
            n_results=10
        )
        ids = results.get('ids')[0]
        prompt = create_icl(ids, train_data)
        # code_str = get_code_program(data[i]['program'])
        facts_str = get_facts(i)
        inst = "Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:"
        input_text = init_prompt + prompt + f"{data[i]['question']}'\nentities = {entities}\nconcepts = {concepts}\nfacts = {facts_str}{inst}\n"
        time_flag = True
        while time_flag:
            try:
                result = generate_output(input_text, model_name)
                if "candidates" in result.keys() and len(result["candidates"]) > 0:
                    output = result["candidates"][0]['content']['parts'][0]['text'].strip()
                    flag = True
                    while flag:
                        parsed_result = output.split('\n')
                        for step_id in range(len(parsed_result)):
                            if "FIND" in parsed_result[step_id] and step_id >= 1 and not "= START()" in parsed_result[step_id - 1]:
                                try:
                                    expression_id = str(int(parsed_result[step_id].split(" = FIND")[0].split('_')[1]) + 1)
                                except:
                                    continue
                                input_text += '\n'.join(parsed_result[:step_id]) + f"\nexpression_{expression_id} = START()\n"
                                break
                        if step_id == len(parsed_result) - 1:
                            flag = False
                        if flag:
                            output = '\n'.join(parsed_result[:step_id]) + f"\nexpression_{expression_id} = START()\n" + generate_output(input_text, api_key, model_name)["candidates"][0]['content']['parts'][0]['text']
                        if not output.startswith('expression') or 'expression()' in output or '```' in output or '!!!!' in output:
                            input_text += f"expression_1 = START()\n"
                            output = "expression_1 = START()\n" + generate_output(input_text, model_name)["candidates"][0]['content']['parts'][0]['text']
                    if output != previous_output:
                        gen_steps.append(output)
                    previous_output = output
                else:
                    output = "NA"
                    gen_steps.append(output)
                f_out = open("Add path of output file", 'w')
                json.dump(gen_steps, f_out, indent=4)
                f_out.close()
                time_flag = False
            except:
                time.sleep(60)
    return gen_steps

gen_steps = run(data, init_prompt)