from entity_typing_framework.EntityTypingNetwork_classes.projectors import Projector
from torch.nn import Linear
import torch

class ALIGNIEProjector(Projector):
  def __init__(self, name, type2id, type_number, input_dim, return_logits=True, **kwargs):
    super().__init__(name, type2id, type_number, input_dim, return_logits, **kwargs)
    # TODO: init
    # TODO: check input_dim == vocab_size
    self.vocab_size = input_dim
    self.label_project = Linear(self.vocab_size, type_number)

    # ??? move into input_encoders.py?
    # # The LM head weights require special treatment only when they are tied with the word embeddings
    # self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

    # Initialize weights and apply final processing (TODO: necessary?)
    self.post_init()

    # from run.py of the original code
    self.init_project(node_id_list, output_num=len(node_id_list), new_instances=new_instances)

  # NOTE: KENN will override this by calling self.ke(super())
  def project_input(self, input_representation : torch):
    # input_representation.shape = (batch_size, vocab_size)
    batch_size = input_representation.shape[0]
    input_representation = input_representation.softmax(dim=-1) # softmax 1...

    input_representation = input_representation.view(batch_size, self.vocab_size)
    # score = grad_multiply(score, lambd=lamb) # is useless with lambd=1 (check original code)
    projected_input = self.label_project(input_representation)
    
    
    # useless??? the loss will be computed in main_module --> eventually move this in get_output_for_loss()
    # log_score_prob = torch.log(torch.softmax(projected_input, dim=-1))
    # if len(masked_label.shape) == 1 or (len(masked_label.shape) == 2 and masked_label.shape[1] == 1):
    #   masked_label = masked_label.view(-1)
    #   masked_label = torch.one_hot(masked_label, num_classes=log_score_prob.shape[1]).to(torch.float32)
    # masked_lm_loss = loss_fct(log_score_prob, masked_label)

    # TODO
    return projected_input

  def classify(self, projected_input : torch):
    # useless???
    # return MaskedLMOutput(loss=masked_lm_loss,
    #         logits=score,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )

    batch_size = projected_input.shape[0]
    return torch.softmax(projected_input.view(batch_size, 1, -1), dim=-1) # ...softmax 2


  def init_project(self, node_id_list): # , output_num, new_instances=None):
    # TODO: implement function to simulate read_file_and_hier() ??
    # problem: node_id_list requires the use of tokenizer
    # use type_encoder?
    # use shared parameter type2label? type2token_id?

    with torch.no_grad(): # why?
      for i in range(len(node_id_list)):
        self.label_project.weight[i] = torch.full((self.vocab_size,), \
            1 / self.type_number)
      for i in range(len(node_id_list)):
        node_id = node_id_list[i]
        for j in range(self.type_number):
          if i == j:
            self.label_project.weight[j][node_id] = 1.0
          else:
            self.label_project.weight[j][node_id] = 0.0
      # USELESS???
      # if new_instances is not None:
      #   for i in range(len(node_id_list)):
      #     node = node_id_list[i]
      #     if node not in new_instances:
      #       continue
      #     for k, v in new_instances[node]:
      #       self.label_project.weight[i][k] += v

  # TODO: is this the right place? use main module to reproduce the steps of this function?
  def predict(
        self,
        input_ids=None,
        masked_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        return_dict=None,
    ):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    with torch.no_grad():
      outputs = self.roberta(
          input_ids,
          attention_mask=attention_mask,
          inputs_embeds=inputs_embeds,
          return_dict=return_dict,
      )
      sequence_output = outputs[0]
      prediction_scores = self.lm_head(sequence_output)

      batch_size,max_len,vocab_size = prediction_scores.shape
      score = prediction_scores.softmax(dim=-1)
  
      masked_ids = masked_ids.repeat(1, 1, vocab_size).reshape(batch_size, 1, vocab_size)
      score = torch.gather(score, 1, masked_ids) # batch x 1 x vocab_size
      return score
