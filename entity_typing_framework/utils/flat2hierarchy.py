import torch

def get_type2id_original(type2id_flat, suffix='/NIL'):
  # remove /NIL to convert to father type
  types = [ k[:-len(suffix)] if k.endswith(suffix) else k for k in type2id_flat.keys() ]
  # add the remaining fathers
  types_unique = set(types)
  for t in types:
    # add father only if t is not top level type
    if len(t[1:].split('/')) > 1:
      father = f"/{'/'.join(t[1:].split('/')[:-1])}"
      types_unique.add(father)

  # create type2id_original
  types = list(types_unique)
  types.sort()
  type2id_original = { t : i for i, t in enumerate(types) }

  return type2id_original

def get_descendants_map(type2id):
  descendants_map = { k : [] for k in type2id.keys() }
  for t_father in type2id.keys():
    for t_descendant in type2id.keys():
      # check if t_k is descendant of t_v
      if t_father != t_descendant and t_father in t_descendant:
        descendants_map[t_father].append(t_descendant)
  return descendants_map

def get_loss_input(encoded_input, type_representation, loss_module):
  encoded_input_flat = encoded_input
  type_representation_original = type_representation
  # convert flat tensors to match the original dataset and include father types in the loss computation
  encoded_input_original = torch.zeros((encoded_input_flat.shape[0], len(loss_module.type2id_original)), device=loss_module.device)
  for t_flat, idx_flat in loss_module.type2id_flat.items():
    # if the type is /*/NIL convert it to father type and assign value
    if t_flat.endswith('/NIL'):
      t = t_flat[:-4]
    else: # the type is shared between flat dataset and original dataset
      t = t_flat
    # copy prediction
    idx = loss_module.type2id_original[t]
    encoded_input_original[:, idx] = encoded_input_flat[:, idx_flat]

  # for each example, for each type assign its value or the max of its descendants
  for i, example in enumerate(encoded_input_original):
    for t, descendants in loss_module.descendants_map_original.items():
      idx_t = loss_module.type2id_original[t]
      max_score = example[idx_t]
      for d in descendants:
        idx_d = loss_module.type2id_original[d]
        score = example[idx_d]
        if score > max_score:
          max_score = example[idx_d]
      # assign max value
      encoded_input_original[i, idx_t] = max_score

    return encoded_input_original, type_representation_original
