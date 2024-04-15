import json
from tqdm import tqdm
import time
import requests
import warnings
warnings.filterwarnings("ignore")


api_key='ADD API KEY'

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
    - The datatypes are as follows:
        - entities: list of entity type
        - value: value of an attribute
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

def STOP(expression: any): 
    ```
    Stop and return the expression
    ```
    return expression

Training Examples:
Training Example 1:
question = "what form of government does afghanistan have?"
entities = ['Afghanistan']
facts = [
{'entity': 'Afghanistan', 'relation': 'form_of_government'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Afghanistan', expression_1) 
expression_1 = RELATE('form of government', expression_1) 
expression_1 = STOP(expression_1)

Training Example 2:
question = "what book did charles darwin write on evolution?"
entities = ['Charles Darwin']
facts = [
{'entity': 'Charles Darwin', 'relation': 'works_written'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Charles Darwin', expression_1)
expression_1 = RELATE('works written', expression_1)
expression_1 = FILTERSTR('notable types', 'Book', expression_1)
expression_1 = FILTERSTR('subjects', 'Evolution', expression_1)
expression_1 = STOP(expression_1)

Training Example 3:
question = "what kind of money does the uk use?"
entities = ['United Kingdom']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('United Kingdom', expression_1)
expression_1 = RELATE('currency used', expression_1)
expression_1 = STOP(expression_1)

Training Example 4:
question = "who is facebook's founder?"
entities = ['Facebook, Inc.']
facts = [
{'entity': 'Facebook, Inc.', 'relation': 'name'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Facebook', expression_1)
expression_1 = RELATE('founders', expression_1)
expression_1 = STOP(expression_1)

Training Example 5:
question = "what is mexico city time zone?"
entities = ['Mexico City']
facts = [
{'entity': 'Mexico City', 'relation': 'time_zones'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Mexico City', expression_1)
expression_1 = RELATE('time zones', expression_1)
expression_1 = STOP(expression_1)

Training Example 6:
question = "what kind of money does germany?"
entities = ['Germany']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Germany', expression_1)
expression_1 = RELATE('currency used', expression_1)
expression_1 = STOP(expression_1)

Training Example 7:
question = "where was nicki minaj born?"
entities = ['Nicki Minaj']
facts = [
{'entity': 'Nicki Minaj', 'relation': 'date_of_birth'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Nicki Minaj', expression_1)
expression_1 = RELATE('place of birth', expression_1)
expression_1 = STOP(expression_1)

Training Example 8:
question = "where does liz mcclarnon live?"
entities = ['Liz McClarnon']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Liz McClarnon', expression_1)
expression_1 = RELATE('places lived', expression_1)
expression_1 = RELATE('location', expression_1)
expression_1 = STOP(expression_1)

Training Example 9:
question = "who is ruling tunisia now?"
entities = ['Tunisia']
facts = [
{'entity': 'Tunisia', 'relation': 'name'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Tunisia', expression_1)
expression_1 = RELATE('governing officials', expression_1)
expression_1 = RELATE('office holder', expression_1)
expression_1 = FILTERSTR('basic title', 'Acting President', expression_1)
expression_1 = STOP(expression_1)

Training Example 10:
question = "who plays jacob black in the twilight movies?"
entities = [''Twilight']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Twilight', expression_1)
expression_1 = RELATE('starring', expression_1)
expression_1 = RELATE('actor', expression_1)
expression_1 = FILTERSTR('character', 'Jacob Black', expression_1)
expression_1 = STOP(expression_1)

Training Example 11:
question = "what countries are on the mediterranean sea?"
entities = ['Mediterranean Sea']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Mediterranean Sea', expression_1)
expression_1 = RELATE('adjoins', expression_1)
expression_1 = RELATE('adjoins', expression_1)
expression_1 = FILTERSTR('notable types', 'Country', expression_1)
expression_1 = STOP(expression_1)

Training Example 12:
question = "what year did tut became king?"
entities = ['Tutankhamun']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Tutankhamun', expression_1)
expression_1 = RELATE('reign', expression_1)
expression_1 = RELATE('tart', expression_1)
expression_1 = STOP(expression_1)


Test Example:
question = "'''


def get_facts(idx, facts):
    facts_str = "[\n"
    for line in range(len(facts)):
        if facts[line] == f'idx: {idx}\n':
            num_triples = int(facts[line + 3].split('TRIPLES: ')[1])
            for t in range(num_triples):
                facts_str += facts[line + 4 + t][:-1] + ",\n"
            if num_triples > 0:
                facts_str = facts_str[:-2] + '\n'
                facts_str += ']\n\n'
            else:
                facts_str = '[]\n\n'
            break
    return facts_str

f = open("../data/webqsp/WebQSP.test.json")
data = json.load(f)['Questions']
f.close()

f1 = open("Add path of facts file")
facts = f1.readlines()
# print(facts)
f1.close()


def run(dataset, init_prompt):
    model_name = "gemini-pro"
    gen_steps_1 = []
    gen_steps_2 = []
    gen_steps_3 = []
    for i in tqdm(range(1261,len(dataset))):
        output = ""
        if (i+1)%50 == 0:
            time.sleep(60)
        ques = dataset[i]['RawQuestion']
        entities = dataset[i]['Parses'][0]['TopicEntityName']
        facts_str = get_facts(i, facts)
        kw_info = f'entities = {entities}\n' + f'facts = {facts_str}'
        inst = "Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:"
        input_text = init_prompt + ques + '"\n'+ kw_info + inst
        time_flag = True
        while time_flag:
            try:
                result = generate_output(input_text, model_name)
                if "candidates" in result.keys() and len(result["candidates"]) > 0:
                    output = result["candidates"][0]['content']['parts'][0]['text'].strip()
                    flag = True
                    while flag:
                        output = output.strip()
                        parsed_result = output.split('\n')
                        try:
                            for step_id in range(len(parsed_result)):
                                if "FIND" in parsed_result[step_id] and step_id >= 1 and not " = START()" in parsed_result[step_id - 1]:
                                    expression_id = str(int(parsed_result[step_id].split(" = FIND")[0].split('_')[1]) + 1)
                                    input_text += '\n'.join(parsed_result[:step_id]) + f"\nexpression_{expression_id} = START()\n"
                                    break
                                elif "FIND" in parsed_result[step_id] and step_id == 0:
                                    expression_id = str(int(parsed_result[step_id].split(" = FIND")[0].split('_')[1]) + 1)
                                    input_text += f"expression_1 = START()\n"
                                    break
                            if step_id == len(parsed_result) - 1:
                                flag = False
                            if flag:
                                output = '\n'.join(parsed_result[:step_id]) + f"\nexpression_{expression_id} = START()\n" + generate_output(input_text, model_name)["candidates"][0]['content']['parts'][0]['text']
                            if not output.startswith('expression') or 'expression()' in output or '```' in output:
                                input_text += f"expression_1 = START()\n"
                                output = "expression_1 = START()\n" + generate_output(input_text, model_name)["candidates"][0]['content']['parts'][0]['text']
                        except:
                            flag = False
                    gen_steps_1.append(output)
                    if len(result["candidates"]) == 3:
                        output = result["candidates"][1]['output'].strip()
                        gen_steps_2.append(output)
                        output = result["candidates"][2]['output'].strip()
                        gen_steps_3.append(output)
                    if len(result["candidates"]) == 2:
                        output = result["candidates"][1]['content']['parts'][0]['text'].strip()
                        gen_steps_2.append(output)
                    else:
                        gen_steps_2.append("NA")
                        gen_steps_3.append("NA")
                else:
                    gen_steps_1.append("NA")
                    gen_steps_2.append("NA")
                    gen_steps_3.append("NA")
                f_out = open("./output_webqsp/out_webqsp_gemini_full_from_1261_new.json", 'w')
                json.dump(gen_steps_1, f_out, indent=4)
                f_out.close()
                time_flag = False
            except:
                time.sleep(60)

    return gen_steps_1, gen_steps_2, gen_steps_3

gen_steps_1, gen_steps_2, gen_steps_3 = run(data, init_prompt)
