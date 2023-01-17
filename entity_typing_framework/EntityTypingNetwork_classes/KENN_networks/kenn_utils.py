from itertools import combinations
import pickle
import os
from treelib import Tree

def create_tree(labels, label2pred = False):
    tree = Tree()
    root = 'thing'
    tree.create_node(root,root)
    for label in sorted(labels): 
        # split levels
        splitted = label[1:].split('/')
        if len(splitted) == 1:
            # new first level node
            parent = root
        else:
            # init parent node
            parent = ''
            # convert to kenn predicate (must start with uppercase letter)
            if label2pred:
                parent = 'P'
            parent += '/' + '/'.join(splitted[:-1])
            if not tree.contains(parent):
                tree.create_node(parent,parent,root)
        # convert to kenn predicate (must start with uppercase letter)
        if label2pred:
            label = 'P' + label
        tree.create_node(label,label,parent)
    return tree

### KENN CONSTRAINTS ###
# mode in ['bottom_up','top_down','hybrid','hybrid_in','hybrid_out','bottom_up_skip', 'top_down_skip']
def generate_constraints(types_list, mode, filepath = None, weight='_'):
    # create ontology tree
    tree = create_tree(types_list, label2pred = True)
    # generate predicate list
    predicates = generate_predicates(types_list)
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
        raise Exception(f'Mode {mode} not handled...')
    # print number of clauses
    print(clauses.count('\n'), 'clauses created')
    # create kb
    kb = predicates + '\n\n' + clauses

    return save_kb(filepath, kb)

def generate_constraints_incremental(all_types, new_types, filepath = None, weight='_', mode='top_down'):
    # create ontology tree from all_types
    tree = create_tree(all_types, label2pred = True)
    # create specialization clauses
    clauses = ''
    fathers = []
    for t in new_types:
        father = tree.parent('P'+t).identifier
        # check if a new clause is needed
        if father not in fathers: 
            fathers.append(father)
            # get subtree where the father of t is the root
            subtree = tree.subtree(father)
            # generate constraints
            if mode == 'bottom_up':
                clauses = generate_bottom_up(subtree, weight)
            elif mode == 'top_down':
                clauses = generate_top_down(subtree, weight)
            elif mode == 'hybrid':
                clauses += generate_hybrid(subtree, weight)
            else:
                raise Exception(f'Mode {mode} not handled...')
    # generate predicate list
    predicates = generate_predicates(all_types)
    # print number of clauses
    print(clauses.count('\n'), 'specialization clauses created')
    # create kb
    kb = predicates + '\n\n' + clauses

    return save_kb(filepath, kb)

def label2pred(t):
    return f'P{t}'

# def generate_constraints_cross_dataset(all_types, new_types, filepath = None, weight='_', tgt2src={}):
#     # create ontology tree from all_types
#     types_src = list(tgt2src.values())
#     # create cross dataset clauses
#     direct_clauses = [f'{weight}:{label2pred(t_src)},{label2pred(t_dst)}' for t_dst, t_src in tgt2src.items() if t_dst]
#     # TODO: add trasversal (with tgt2src_clause parameter it would replace the line above)
#     new_types_unmapped = list(set(new_types) - set(tgt2src.keys()))
#     negative_clauses = [ f'{weight}:n{label2pred(t_src)},n{label2pred(t_dst)}' for t_src in types_src for t_dst in new_types_unmapped]
#     # generate predicate list
#     # TODO: check order
#     predicates = generate_predicates(all_types)
#     # print number of clauses
#     clauses = direct_clauses + negative_clauses
#     print(clauses.count('\n'), 'cross-dataset clauses created')
#     # create kb
#     kb = predicates + '\n\n' + '\n'.join(clauses) + '\n'

#     return save_kb(filepath, kb)

def generate_constraints_cross_dataset(all_types, new_types, filepath = None, weight='_', tgt2src={}):
    # create ontology tree from all_types
    types_src = list(tgt2src.values())
    # create cross dataset clauses
    direct_clauses = [f'{weight}:{label2pred(t_src)},{label2pred(t_dst)}' for t_dst, t_src in tgt2src.items() if t_dst]
    # TODO: add trasversal (with tgt2src_disjoint parameter it would replace the line above)
    new_types_unmapped = list(set(new_types) - set(tgt2src.keys()))
    negative_clauses = [ f'{weight}:n{label2pred(t_src)},n{label2pred(t_dst)}' for t_src in types_src for t_dst in new_types_unmapped]
    # generate predicate list
    # TODO: check order
    predicates = generate_predicates(all_types)
    # print number of clauses
    clauses = direct_clauses + negative_clauses
    print(clauses.count('\n'), 'cross-dataset clauses created')
    # create kb
    kb = predicates + '\n\n' + '\n'.join(clauses) + '\n'

    return save_kb(filepath, kb)


def save_kb(filepath, kb):
    if filepath:
        folder_path = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(filepath, 'w') as f:
            f.write(kb)
        return kb

def generate_predicates(types_list):
    return ','.join([f'{label2pred(t)}' for t in types_list])


def generate_bottom_up(tree, weight):
    clauses = ""
    if tree.root == 'thing':
        roots = [x.identifier for x in tree.children(tree.root)]
    else: # specialization
        roots = [tree.root]
    # iterate over each tree of the forest
    for root in roots:
        # get single tree
        subtree = tree.subtree(root)
        # get all descendants
        descendants = subtree.filter_nodes(lambda x : subtree.depth(x) != 0)
        # create a subtype -> supertype rule for each node
        for descendant in descendants:
            parent = subtree.parent(descendant.identifier)
            clauses += f"{weight}:n{descendant.identifier},{parent.identifier}\n"
    return clauses

def generate_top_down(tree, weight):
    clauses = ""
    if tree.root == 'thing':
        roots = [x.identifier for x in tree.children(tree.root)]
    else: # specialization
        roots = [tree.root]
    # iterate over each tree of the forest
    for root in roots:
        # get single tree
        subtree = tree.subtree(root)
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
