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
Training Example 1:
question = "How many researchers are the occupation of Aristotle or practice motivational speaking?"
entities = ['Aristotle', 'motivational speaking']
concepts = ['researcher']
facts = [
{'entity': 'Aristotle', 'relation': 'occupation'},
{'concept': 'researcher', 'relation': 'occupation'},
{'concept': 'researcher', 'relation': 'field of this occupation'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Aristotle', expression_1) 
expression_1 = RELATE('occupation', expression_1) 
expression_1 = FILTERCONCEPT('researcher', expression_1)
expression_2 = START()
expression_2 = FIND('motivational speaking', expression_2) 
expression_2 = RELATE('practiced by', expression_2) 
expression_2 = FILTERCONCEPT('researcher', expression_2) 
expression_3 = OR(expression_1, expression_2) 
expression_3 = COUNT(expression_3)
expression_3 = STOP(expression_3)

Training Example 2:
question = "What is the connection between A Serious Man to Ireland (the one whose nominal GDP is 239389340720.488 United States dollar)?"
entities = ['A Serious Man', 'Ireland']
concepts = None
facts = [
{'entity': 'Ireland', 'attribute': 'PPP GDP per capita'},
{'entity': 'Ireland', 'attribute': 'GDP (PPP)'},
{'entity': 'Ireland', 'attribute': 'nominal GDP per capita'},
{'entity': 'Ireland', 'attribute': 'nominal GDP'},
{'entity': 'Ireland', 'relation': 'currency'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('A Serious Man', expression_1)
expression_2 = START()
expression_2 = FIND('Ireland', expression_2)
expression_2 = FILTERNUM('nominal GDP', '239389340720.488 United States dollar', '=', expression_2)
expression_3 = QUERYRELATION(expression_1, expression_2)
expression_3 = STOP(expression_3)

Training Example 3:
question = "Which first-level administrative country subdivision established post-1829 covers the biggest area?"
entities = None
concepts = ['first-level administrative country subdivision']
facts = [
{'concept': 'first-level administrative country subdivision', 'relation': 'operating area'},
{'concept': 'first-level administrative country subdivision', 'attribute': 'area'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FINDALL(expression_1)
expression_1 = FILTERYEAR('inception', 1829, '>', expression_1)
expression_1 = FILTERCONCEPT('first-level administrative country subdivision', expression_1)
expression_1 = SELECTAMONG('area', 'largest', expression_1)
expression_1 = STOP(expression_1)

Training Example 4:
question = "What is the ISNI of John Broome (the one born in 1738-01-01)?"
entities = ['John Broome']
concepts = None
facts = [
{'entity': 'John Broome', 'attribute': 'ISNI'},
{'entity': 'John Broome', 'relation': 'place of birth'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('John Broome', expression_1)
expression_1 = FILTERDATE('date of birth', '1738-01-01', '=', expression_1)
expression_1 = QUERYATTR('ISNI', expression_1)
expression_1 = STOP(expression_1)

Training Example 5:
question = "Does the sovereign state that has a diplomatic relation with Malaysia (the subject of this statement is East Timor–Malaysia relations), have the CIVICUS Monitor country entry of saint-lucia?"
entities = ['Malaysia']
concepts = ['sovereign state']
facts = [
{'entity': 'Malaysia', 'attribute': 'CIVICUS Monitor country entry'},
{'entity': 'Malaysia', 'relation': 'diplomatic relation', 'qualifier': 'statement is subject of'},
{'entity': 'Malaysia', 'relation': 'country'},
{'entity': 'Malaysia', 'relation': 'country for sport'},
{'concept': 'sovereign state', 'relation': 'country', 'qualifier': 'statement disputed by'},
{'concept': 'sovereign state', 'relation': 'diplomatic relation', 'qualifier': 'statement is subject of'},
{'concept': 'sovereign state', 'relation': 'country of origin'},
{'concept': 'sovereign state', 'relation': 'country for sport'},
{'concept': 'sovereign state', 'relation': 'main subject'},
{'concept': 'sovereign state', 'attribute': 'CIVICUS Monitor country entry'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Malaysia', expression_1)
expression_1 = RELATE('diplomatic relation', expression_1)
expression_1 = QFILTERSTR('statement is subject of', 'East Timor–Malaysia relations', expression_1)
expression_1 = FILTERCONCEPT('sovereign state', expression_1)
expression_1 = QUERYATTR('CIVICUS Monitor country entry', expression_1)
expression_1 = VERIFYSTR('saint-lucia', expression_1)
expression_1 = STOP(expression_1)

Training Example 6:
question = "What is the umber of episodes in TV series with Twitter username ThomasFriends (the subscription number of this statement is 15947)?"
entities = None
concepts = ['television series']
facts = [
{'concept': 'television series', 'relation': 'part of the series'},
{'concept': 'television series', 'relation': 'series spin-off'},
{'concept': 'television series', 'attribute': 'number of episodes'},
{'concept': 'television series', 'attribute': 'Twitter username', 'qualifier': 'number of subscribers'},
{'concept': 'television series', 'attribute': 'Instagram username'},
{'concept': 'television series', 'attribute': 'Twitter hashtag'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FINDALL(expression_1)
expression_1 = FILTERSTR('Twitter username', 'ThomasFriends', expression_1)
expression_1 = QFILTERNUM('number of subscribers', 15947, '=', expression_1)
expression_1 = FILTERCONCEPT('television series', expression_1)
expression_1 = QUERYATTR('number of episodes', expression_1)
expression_1 = STOP(expression_1)

Training Example 7:
question = "When was born the person that was nominated for Tony Award for Best Actor in a Musical in 1967?"
entities = ['Tony Award for Best Actor in a Musical']
concepts = ['human']

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Tony Award for Best Actor in a Musical', expression_1)
expression_1 = RELATE('nominated for', expression_1)
expression_1 = QFILTERYEAR('point in time', 1967, '=', expression_1)
expression_1 = FILTERCONCEPT('human', expression_1)
expression_1 = QUERYATTR('date of birth', expression_1)
expression_1 = STOP(expression_1)

Training Example 8:
question = "Does Pierce County that is located in Washington or Grays Harbor County have less area?"
entities = ['Washington', 'Pierce County', 'Grays Harbor County']
concepts = None
facts = [
{'entity': 'Pierce County', 'attribute': 'area'},
{'entity': 'Washington', 'attribute': 'area'},
{'entity': 'Washington', 'relation': 'headquarters location'},
{'entity': 'Grays Harbor County', 'attribute': 'area'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Washington', expression_1)
expression_1 = RELATE('located in the administrative territorial entity', expression_1)
expression_2 = START()
expression_2 = FIND('Pierce County', expression_2)
expression_3 = AND(expression_1, expression_2)
expression_4 = START()
expression_4 = FIND('Grays Harbor County', expression_4)
expression_5 = SELECTBETWEEN('area', 'less', expression_3, expression_4)
expression_5 = STOP(expression_5)

Training Example 9:
question = "Is the nominal GDP of Guinea-Bissau over 69000000 United States dollars on the date 1996-01-01?"
entities = ['Guinea-Bissau']
concepts = None
facts = [
{'entity': 'Guinea-Bissau', 'attribute': 'PPP GDP per capita'},
{'entity': 'Guinea-Bissau', 'attribute': 'GDP (PPP)'},
{'entity': 'Guinea-Bissau', 'attribute': 'nominal GDP per capita'},
{'entity': 'Guinea-Bissau', 'attribute': 'nominal GDP'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Guinea-Bissau', expression_1)
expression_1 = QUERYATTRUNDERCONDITION('nominal GDP', 'point in time', '1996-01-01', expression_1)
expression_1 = VERIFYNUM('69000000 United States dollar', '>', expression_1)
expression_1 = STOP(expression_1)

Training Example 10:
question = "Which university has fewer students, George Washington University or University of Hamburg?"
entities = ['George Washington University', 'University of Hamburg']
concepts = None
facts = [
{'entity': 'George Washington University', 'attribute': 'students count'},
{'entity': 'University of Hamburg', 'attribute': 'students count'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('George Washington University', expression_1)
expression_2 = START()
expression_2 = FIND('University of Hamburg', expression_2)
expression_3 = SELECTBETWEEN('students count', 'less', expression_1, expression_2)
expression_3 = STOP(expression_3)

Training Example 11:
question = "Was James Hetfield born on 1967-02-10?"
entities = ['James Hetfield']
concepts = None
facts = [
{'entity': 'James Hetfield', 'attribute': 'date of birth'},
{'entity': 'James Hetfield', 'relation': 'place of birth'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('James Hetfield', expression_1)
expression_1 = QUERYATTR('date of birth', expression_1)
expression_1 = VERIFYDATE('1967-02-10', '=', expression_1)
expression_1 = STOP(expression_1)

Training Example 12:
question = "What language is http://sydneyolympicfc.com.au, the official website of an association football club whose Instagram username is sofc1957?"
entities = None
concepts = ['association football club']
facts = [
{'concept': 'association football club', 'relation': 'official color'},
{'concept': 'association football club', 'attribute': 'official website', 'qualifier': 'language of work or name'},
{'concept': 'association football club', 'attribute': 'Instagram username'},
{'concept': 'association football club', 'attribute': 'official name'},
{'concept': 'association football club', 'attribute': 'Twitter username', 'qualifier': 'language of work or name'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FINDALL(expression_1)
expression_1 = FILTERSTR('Instagram username', 'sofc1957', expression_1)
expression_1 = FILTERCONCEPT('association football club', expression_1)
expression_1 = QUERYATTRQUALIFIER('official website', 'http://sydneyolympicfc.com.au/', 'language of work or name', expression_1)
expression_1 = STOP(expression_1)

Training Example 13:
question = "For which work was Armando Iannucci nominated for an Academy Award for Best Writing, Adapted Screenplay?"
entities = ['Armando Iannucci', 'Academy Award for Best Writing, Adapted Screenplay']
concepts = None
facts = [
{'entity': 'Academy Award for Best Writing, Adapted Screenplay', 'relation': 'nominated for'},
{'entity': 'Academy Award for Best Writing, Adapted Screenplay', 'relation': 'nominated for', 'qualifier': 'for work'},
{'entity': 'Armando Iannucci', 'relation': 'nominated for', 'qualifier': 'for work'}
]

Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:
expression_1 = START()
expression_1 = FIND('Armando Iannucci', expression_1)
expression_2 = START()
expression_2 = Find('Academy Award for Best Writing, Adapted Screenplay', expression_2)
expression_3 = QUERYRELATIONQUALIFIER('nominated for', 'statement is subject of', expression_1, expression_2)
expression_3 = STOP(expression_3)


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

f = open("../data/KQAPro.IID/val.json")
data = json.load(f)
f.close()


f1 = open("Add path of facts file retrieved using Question-specific subgraph information")
facts = f1.readlines()
f1.close()

def run(dataset, init_prompt):
    model_name = "gemini-pro"
    gen_steps_1 = []
    gen_steps_2 = []
    gen_steps_3 = []
    for i in tqdm(range(0,len(data))):
        output = ""
        if (i+1)%50 == 0:
            time.sleep(60)
        entities, concepts = get_entities_concepts(dataset[i]['program'])
        facts_str = get_facts(i, facts)
        kw_info = f'entities = {entities}\n' + f'concepts = {concepts}\nfacts = {facts_str}\n'
        inst = "Make sure to validate the datatype of the parameter before selecting a function using assert statements provided in each function. Generate Find function for each entity in the entities list. Given the facts provided, the steps to solve this question are:"
        input_text = init_prompt + dataset[i]['question'] + '"\n'+ kw_info + inst

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
                f_out = open(f"Add path of output file", 'w')
                json.dump(gen_steps_1, f_out, indent=4)
                f_out.close()
                
                time_flag = False
            except:
                time.sleep(60)

    return gen_steps_1, gen_steps_2, gen_steps_3

gen_steps_1, gen_steps_2, gen_steps_3 = run(data, init_prompt)