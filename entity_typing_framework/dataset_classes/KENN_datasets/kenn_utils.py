from itertools import combinations
import pickle
import os

### KENN CONSTRAINTS ###
# mode in ['bottom_up','top_down','hybrid','hybrid_in','hybrid_out','bottom_up_skip', 'top_down_skip']
def generate_constraints(tree, mode, filepath, weight='_'):
    # generate predicate list
    predicates = generate_predicates(tree)
    # generate constraints
    if mode == 'bottom_up':
        clauses = generate_bottom_up(tree, weight)
    elif mode == 'top_down':
        clauses = generate_top_down(tree, weight)
    elif mode == 'hybrid':
        clauses = generate_hybrid(tree, weight)
    elif mode == 'hybrid_in':
        if tree.depth() == 3:
            clauses = generate_hybrid_in(tree, weight)
        else:
            print('The hierarchy must have 3 levels')
            return None
    elif mode == 'hybrid_out':
        if tree.depth() == 3:
            clauses = generate_hybrid_out(tree, weight)
        else:
            print('The hierarchy must have 3 levels')
            return None
    elif mode == 'bottom_up_plus':
        clauses = generate_bottom_up(tree, weight)
        clauses += generate_horizontal(tree, weight, 'top_level')
    elif mode == 'top_down_plus':
        clauses = generate_top_down(tree, weight)
        clauses += generate_horizontal(tree, weight, 'top_level')
    elif mode == 'hybrid_plus':
        clauses = generate_hybrid(tree, weight)
        clauses += generate_horizontal(tree, weight, 'top_level')
    elif mode == 'bottom_up_skip':
        pass
    elif mode == 'top_down_skip':
        pass
    else:
        print('Mode not handled...')
        return None
    # print number of clauses
    print(clauses.count('\n'), 'clauses created')
    # create kb
    kb = predicates + '\n\n' + clauses
    
    folder_path = '/'.join(filepath.split('/')[:-1])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(filepath, 'w') as f:
        f.write(kb)
    return kb

def generate_predicates(tree):
    predicates = []
    for n in tree.filter_nodes(lambda x : tree.depth(x) != 0):
        predicates.append(n.identifier)
    return ','.join(predicates)


def generate_bottom_up(tree, weight):
    clauses = ""
    # iterate over each tree of the forest
    for root in tree.children(tree.root):
        # get single tree
        subtree = tree.subtree(root.identifier)
        # get all descendants
        descendants = subtree.filter_nodes(lambda x : subtree.depth(x) != 0)
        # create a subtype -> supertype rule for each node
        for descendant in descendants:
            parent = subtree.parent(descendant.identifier)
            clauses += f"{weight}:n{descendant.identifier},{parent.identifier}\n"
    return clauses

def generate_top_down(tree, weight):
    clauses = ""
    # iterate over each tree of the forest
    for root in tree.children(tree.root):
        # get single tree
        subtree = tree.subtree(root.identifier)
        # get all the internal nodes
        internal_nodes = subtree.filter_nodes(lambda x : x not in subtree.leaves())
        # create a subtype -> supertype rule for each node
        for internal_node in internal_nodes:
            children = subtree.children(internal_node.identifier)
            clauses += f"{weight}:n{internal_node.identifier}"
            for child in children:
                clauses += f",{child.identifier}"
            clauses += '\n'
    return clauses

def generate_hybrid(tree, weight):
    clauses = generate_bottom_up(tree, weight) 
    clauses += generate_top_down(tree, weight)
    return clauses

def generate_horizontal(tree, weight, level='top_level'):
    if level == 'top_level':
        clauses = ""
        # get all the top_level nodes
        nodes = [x.identifier for x in tree.children(tree.root)]
        # create positive clause: nA => (B v C) becomes A v B v C
        clauses = f"{weight}:{','.join(nodes)}"
        clauses += '\n'
        # create pairs of disjunctions: A => nB, A => nC, ... become nA v nB, nB v nC, nA v nC
        for pair in combinations(nodes,2):    
            clauses += f"{weight}:n{pair[0]},n{pair[1]}\n"
        return clauses
    else:
        return ''
    

def generate_hybrid_in(tree, weight):
    clauses = ""
    # iterate over each tree of the forest
    for root in tree.children(tree.root):
        # get single tree
        subtree = tree.subtree(root.identifier)
        # get all descendants
        descendants = subtree.filter_nodes(lambda x : subtree.depth(x) != 0)
        for descendant in descendants:
            parent = subtree.parent(descendant.identifier)
            if tree.depth(descendant.identifier) == 3:
                # bottom_up
                clauses += f"{weight}:n{descendant.identifier},{parent.identifier}\n"
        
        # top_down
        children = subtree.children(root.identifier)
        clauses += f"{weight}:n{descendant.identifier}"
        for child in children:
            clauses += f",{child.identifier}"
        clauses += '\n'

    return clauses

def generate_hybrid_out(tree, weight):
    clauses = ""
    # iterate over each tree of the forest
    for root in tree.children(tree.root):
        # get single tree
        subtree = tree.subtree(root.identifier)
        # get all descendants
        descendants = subtree.filter_nodes(lambda x : subtree.depth(x) != 0)
        for descendant in descendants:
            parent = subtree.parent(descendant.identifier)
            if tree.depth(descendant.identifier) == 2:
                # bottom_up
                clauses += f"{weight}:n{descendant.identifier},{parent.identifier}\n"
                # top_down
                children = subtree.children(descendant.identifier)
                top_down_clause = ""
                for child in children:
                    top_down_clause += f",{child.identifier}"
                if top_down_clause != "":
                    clauses += f"{weight}:n{descendant.identifier}" + top_down_clause
                    clauses += '\n'

    return clauses

### CLAUSE/WEIGHTS UTILS  ###
# map weights and clauses
def get_weighted_clauses(model):
  weighted_rules = {}
  for clause in model.ke.knowledge_enhancer.children():
    rule = clause.clause_string.replace('\n','').replace('nP/','/').replace('P/','/')
    rule = rule.split(',')[0] + ' => ' + ' v '.join(rule.split(',')[1:])
    weighted_rules[rule] = round(clause.clause_weight.item(), 4)
  return weighted_rules

def get_clauses_list(model):
  clauses = []
  for clause in model.ke.knowledge_enhancer.children():
    rule = clause.clause_string.replace('\n','').replace('nP/','/').replace('P/','/')
    rule = rule.split(',')[0] + ' => ' + ' v '.join(rule.split(',')[1:])
    clauses.append(rule)
  return clauses

def get_clauses_list_from_file(kb_path):
  clauses = []
  with open(kb_path, 'r') as f:
    # ignore intestation
    f.readline()
    # ignore empty line
    f.readline()
    # read rules
    for i, line in enumerate(f.readlines()):
      rule = line.split(':')[1].replace('\n','').replace('nP/','/').replace('P/','/')
      rule = rule.split(',')[0] + ' => ' + ' v '.join(rule.split(',')[1:])
      clauses.append(rule)
  return clauses

def save_weighted_clauses(filename, clauses, format='txt'):
  filepath = filename + '.' + format
  if format == 'txt':
    with open(filepath, 'w') as f:
      for clause, weight in clauses.items():
        f.write(f"{weight} : {clause}\n")
  elif format == 'pkl':
    with open(filepath, 'wb') as f:
      pickle.dump(clauses, f)

def load_weighted_clauses(filepath):
  with open(filepath, 'rb') as f:
        return pickle.load(f) 

def get_custom_clause_weights(kb_path, stats, threshold, normal_weight, decreased_weight):
    # read clauses
    clauses = get_clauses_list_from_file(kb_path)
    # filter antecedent labels
    antecedents = [ clause.split(' => ')[0] for clause in clauses ]
    # assign weights
    initial_clause_weights = []
    for i, label in enumerate(antecedents):
        w = decreased_weight if stats[label] > threshold else normal_weight
        initial_clause_weights.append(w)
    return initial_clause_weights
