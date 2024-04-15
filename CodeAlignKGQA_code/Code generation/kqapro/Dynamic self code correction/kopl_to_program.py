import json
import networkx as nx
import signal
import time

def get_code_program(program):    
    code_program = ""
    expression_id = 0
    expression_id_dict = {}
    for j in range(len(program)):
        function = program[j]['function']
        inputs = program[j]['inputs']
        dep = program[j]['dependencies']
        if function == 'Find':
            expression_id += 1
            code_program += f"expression_{expression_id} = START()\nexpression_{expression_id} = FIND('{inputs[0]}', expression_{expression_id})\n"
        elif function == 'FindAll':
            expression_id += 1
            code_program += f"expression_{expression_id} = START()\nexpression_{expression_id} = FINDALL(expression_{expression_id})\n"
        elif function == 'Count':
            code_program += f"expression_{expression_id} = COUNT(expression_{expression_id})\n"
        elif function == "Relate":
            code_program += f"expression_{expression_id} = RELATE('{inputs[0]}', expression_{expression_id})\n"
        elif function == "FilterConcept":
            code_program += f"expression_{expression_id} = FILTERCONCEPT('{inputs[0]}', expression_{expression_id})\n"
        elif function == "FilterStr":
            code_program += f"expression_{expression_id} = FILTERSTR('{inputs[0]}', '{inputs[1]}', expression_{expression_id})\n" 
        elif function == "FilterNum":
            code_program += f"expression_{expression_id} = FILTERNUM('{inputs[0]}', '{inputs[1]}', '{inputs[2]}', expression_{expression_id})\n"
        elif function == "FilterYear":
            code_program += f"expression_{expression_id} = FILTERYEAR('{inputs[0]}', '{inputs[1]}', '{inputs[2]}', expression_{expression_id})\n"
        elif function == "FilterDate":
            code_program += f"expression_{expression_id} = FILTERDATE('{inputs[0]}', '{inputs[1]}', '{inputs[2]}', expression_{expression_id})\n"
        elif function == "QFilterStr":
            code_program += f"expression_{expression_id} = QFILTERSTR('{inputs[0]}', '{inputs[1]}', expression_{expression_id})\n"
        elif function == "QFilterNum":
            code_program += f"expression_{expression_id} = QFILTERNUM('{inputs[0]}', '{inputs[1]}', '{inputs[2]}', expression_{expression_id})\n"
        elif function == "QFilterYear":
            code_program += f"expression_{expression_id} = QFILTERYEAR('{inputs[0]}', '{inputs[1]}', '{inputs[2]}', expression_{expression_id})\n"
        elif function == "QFilterDate":
            code_program += f"expression_{expression_id} = QFILTERDATE('{inputs[0]}', '{inputs[1]}', '{inputs[2]}', expression_{expression_id})\n"  
        elif function == "And":
            expression_id += 1
            code_program += f"expression_{expression_id} = AND(expression_{expression_id_dict[dep[0]]}, expression_{expression_id_dict[dep[1]]})\n"
        elif function == "Or":
            expression_id += 1
            code_program += f"expression_{expression_id} = OR(expression_{expression_id_dict[dep[0]]}, expression_{expression_id_dict[dep[1]]})\n"  
        elif function == "QueryRelation":
            expression_id += 1
            code_program += f"expression_{expression_id} = QUERYRELATION(expression_{expression_id_dict[dep[0]]}, expression_{expression_id_dict[dep[1]]})\n"
        elif function == "SelectBetween":
            expression_id += 1
            code_program += f"expression_{expression_id} = SELECTBETWEEN('{inputs[0]}', '{inputs[1]}', expression_{expression_id_dict[dep[0]]}, expression_{expression_id_dict[dep[1]]})\n"
        elif function == "SelectAmong":
            code_program += f"expression_{expression_id} = SELECTAMONG('{inputs[0]}', '{inputs[1]}', expression_{expression_id})\n"
        elif function == "QueryAttr":
            code_program += f"expression_{expression_id} = QUERYATTR('{inputs[0]}', expression_{expression_id})\n"
        elif function == "QueryAttrUnderCondition":
            code_program += f"expression_{expression_id} = QUERYATTRUNDERCONDITION('{inputs[0]}', '{inputs[1]}', '{inputs[2]}', expression_{expression_id})\n"
        elif function == "VerifyStr":
            code_program += f"expression_{expression_id} = VERIFYSTR('{inputs[0]}', expression_{expression_id})\n"
        elif function == "VerifyNum":
            code_program += f"expression_{expression_id} = VERIFYNUM('{inputs[0]}', '{inputs[1]}', expression_{expression_id})\n"   
        elif function == "VerifyYear":
            code_program += f"expression_{expression_id} = VERIFYYEAR('{inputs[0]}', '{inputs[1]}', expression_{expression_id})\n"
        elif function == "VerifyDate":
            code_program += f"expression_{expression_id} = VERIFYDATE('{inputs[0]}', '{inputs[1]}', expression_{expression_id})\n"
        elif function == "QueryAttrQualifier":
            code_program += f"expression_{expression_id} = QUERYATTRQUALIFIER('{inputs[0]}', '{inputs[1]}', '{inputs[2]}', expression_{expression_id})\n"
        elif function == "QueryRelationQualifier":
            # print(inputs)
            # print(dep)
            code_program += f"expression_{expression_id} = QUERYRELATIONQUALIFIER('{inputs[0]}', '{inputs[1]}', expression_{expression_id_dict[dep[0]]}, expression_{expression_id_dict[dep[1]]})\n"
        expression_id_dict[j] = expression_id           

    code_program += f"expression_{expression_id} = STOP(expression_{expression_id})"
    return code_program

def code_program_to_kopl(code_program):
    kopl = ""
    steps = code_program.split('\n')[:-1]
    for step in steps:
        step = step.split(" = ")[1].split('(')[0].lower()
        if step != 'start':
            kopl += f' {step} |'
    kopl = kopl[1:-1]
    return kopl

def kopl_graph(program):
    # print('in kopl graph')
    G = nx.DiGraph()
    node_id = {}
    node_mapping = {} 
    for j in range(len(program)):
        node = program[j]['function']
        if node not in node_id.keys():
            G.add_node(node)
            node_id[node] = 1
            node_mapping[j] = node
        else:
            G.add_node(f'{node}_{node_id[node]}')
            node_id[node] += 1
            node_mapping[j] = f'{node}_{node_id[node]}'

    for j in range(len(program)):
        dependency = program[j]['dependencies']
        if len(dependency) > 0:
            for step_id in dependency:
                start_node = program[step_id]['function']
                start_node_mapped = node_mapping[step_id]
                current_node_mapped = node_mapping[j]
                G.add_edge(start_node_mapped, current_node_mapped)
    for node in G.nodes():
        if G.out_degree(node) == 0:
            root = node
    return G, root

def timeout_handler(num, stack):
    # print("Received SIGALRM")
    raise Exception("COMPLEX")

def get_kopl_distance(program_1, G_2, root_2):
    # print('in kopl distance')
    G_1, root_1 = kopl_graph(program_1)
    # print(G_1.edges())
    # print(G_2.edges())
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1)
    # print('distance calculation')
    try:
        start = time.time()
        distance = nx.graph_edit_distance(G_1, G_2, roots=(root_1, root_2))
        end = time.time()
        print((end-start).total_seconds())
    except:
        distance = 1000
    return distance
