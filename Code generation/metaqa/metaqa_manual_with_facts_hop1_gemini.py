import json
from tqdm import tqdm
import time
import requests
import ast
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

def WHAT(expression: any): 
    ```
    Returns the enitity name
    ```
    return 'WHAT()'

def STOP(expression: any): 
    ```
    Stop and return the expression
    ```
    return expression

Training Examples:
Training Example 1:
question = who was The Grey written by.
entities = ['The Grey']
facts = [
{'entity': 'The Grey', 'relation': 'written_by'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('The Grey', expression_1) 
expression_1 = RELATE('written by', expression_1) 
expression_1 = WHAT(expression_1)
expression_1 = STOP(expression_1)

Training Example 2:
question = what terms are applicable to Krrish 3. 
entities = ['Krrish 3']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Krrish 3', expression_1) 
expression_1 = RELATE('has tags', expression_1)
expression_1 = WHAT(expression_1)
expression_1 = STOP(expression_1)

Training Example 3:
question = what movies are about albert brooks.
entities: ['albert brooks']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('albert brooks', expression_1) 
expression_1 = RELATE('has tags', expression_1)
expression_1 = WHAT(expression_1)
expression_1 = STOP(expression_1)

Training Example 4:
question = who is the author of The Town.
entities: ['The Town']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('The Town', expression_1) 
expression_1 = RELATE('written by', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 5:
question = who acted in Daisies.
entities = ['Daisies']
facts = [
{'entity': 'Daisies', 'relation': 'starred_actors'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Daisies', expression_1) 
expression_1 = RELATE('starred actors', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 6:
question = can you give a few words describing Where the Red Fern Grows.
entities = ['Where the Red Fern Grows']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Where the Red Fern Grows', expression_1) 
expression_1 = RELATE('has tags', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 7:
question = what films did Fisher Stevens direct.
entities = ['Fisher Stevens']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Fisher Stevens', expression_1) 
expression_1 = RELATE('directed by', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 8:
question = who starred in Beautiful Girls.
entities = ['Beautiful Girls']
facts = [
{'entity': 'Beautiful Girls', 'relation': 'starred_actors'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Beautiful Girls', expression_1) 
expression_1 = RELATE('starred actors', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 9:
question = what was the release year of The Curse of the Werewolf. 
entities = ['The Curse of the Werewolf']
facts = [
{'entity': 'The Curse of the Werewolf', 'relation': 'release_year'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('The Curse of the Werewolf', expression_1) 
expression_1 = RELATE('release_year', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 10:
question = which words describe film Twisted.
entities = ['Twisted']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Twisted', expression_1) 
expression_1 = RELATE('has tags', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 11:
question = what is the primary language in the film The Wild Child. 
entities = ['The Wild Child']
facts = [
{'entity': 'The Wild Child', 'relation': 'in_language'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('The Wild Child', expression_1) 
expression_1 = RELATE('language', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 12:
question = who was the director of Millennium Actress.
entities = ['Millennium Actress']
facts = [
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Millennium Actress', expression_1) 
expression_1 = RELATE('directed by', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 13:
question = who acted in Stella.
entities = ['Stella']
facts = [
{'entity': 'Stella', 'relation': 'starred_actors'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Stella', expression_1) 
expression_1 = RELATE('starred actors', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 14:
question = when was the film Life Partners released.
entities = ['Life Partners']
facts = [
{'entity': 'Life Partners', 'relation': 'release_year'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Life Partners', expression_1) 
expression_1 = RELATE('release_year', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)

Training Example 15:
question = what movie did Kevin Tenney write the story for.
entities = ['Kevin Tenney']
facts = [
{'entity': 'Kevin Tenney', 'relation': 'written_by'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Kevin Tenney', expression_1) 
expression_1 = RELATE('written by', expression_1)
expression_1 = WHAT(expression_1) 
expression_1 = STOP(expression_1)


Test Example:
question = "'''


def get_facts(idx, facts):
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


f = open('../data/metaQA/1-hop/vanilla/qa_test.txt')
test_data = f.readlines()
f.close()

data = []
entity_list = []
for i in range(len(test_data)):
    ques = test_data[i].split('\t')[0]
    entity = ques[ques.find("[")+1:ques.find("]")]
    ques = ques.replace('[', '').replace(']', '')
    data.append(ques)
    entity_list.append([entity])


f1 = open("Add path of facts file")
facts = f1.readlines()
f1.close()


def run(dataset, init_prompt):
    model_name = "gemini-pro"
    gen_steps_1 = []
    gen_steps_2 = []
    gen_steps_3 = []

    for i in tqdm(range(0,len(dataset))):
        output = ""
        if (i+1)%50 == 0:
            time.sleep(60)
        ques = dataset[i]
        entities = entity_list[i]
        facts_str = get_facts(i, facts)
        kw_info = f'entities = {entities}\nfacts = {facts_str}'
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
                f_out = open("Add output file path", 'w')
                json.dump(gen_steps_1, f_out, indent=4)
                f_out.close()
                time_flag = False
            except:
                time.sleep(60)

    return gen_steps_1, gen_steps_2, gen_steps_3

gen_steps_1, gen_steps_2, gen_steps_3 = run(data, init_prompt)
