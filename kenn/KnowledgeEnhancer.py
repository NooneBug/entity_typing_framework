import torch
from .ClauseEnhancer import ClauseEnhancer


class KnowledgeEnhancer(torch.nn.Module):

    def __init__(self, predicates: [str], clauses: [str], initial_clause_weight=0.5, save_training_data=False, device=0):
        """Initialize the knowledge base.
        :param predicates: a list of predicates names
        :param clauses: a list of constraints. Each constraint is a string on the form:
        clause_weight:clause
        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).
        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.
        An example:
           _:nDog,Animal
        :param initial_clause_weight: the initial sign to the clause weight. Used if the clause weight is learned.
        """

        super().__init__()
        self.clause_enhancers = []
        self.save_training_data = save_training_data


        for index, clause in enumerate(clauses):
            clause_weight = initial_clause_weight[index] if isinstance(initial_clause_weight, list) else initial_clause_weight
            enhancer = ClauseEnhancer(
                predicates, clause[:-1], float(clause_weight), device=device)
            self.clause_enhancers.append(enhancer)
            self.add_module(f'clause-{index}', enhancer)

    def forward(self, ground_atoms: torch.Tensor) -> (torch.Tensor, [torch.Tensor, torch.Tensor]):
        """Improve the satisfaction level of a set of clauses.
        :param ground_atoms: the tensor containing the pre-activation values of the ground atoms
        :return: final delta values"""

        # scatter_deltas_list will be the list of deltas for each clause
        # e.g. scatter_deltas_list[0] are the deltas relative to the first clause.
        scatter_deltas_list: [torch.Tensor] = []
        light_deltas_list = []
        weights = []
        # TODO: parllelize over clauses
        for enhancer in self.clause_enhancers:
            scattered_delta, delta = enhancer(ground_atoms)
            scatter_deltas_list.append(scattered_delta)
            if self.save_training_data:
                light_deltas_list.append(delta)
                weights.append(enhancer.clause_weight.item())

        deltas_data = [light_deltas_list, weights]
        # The sum can be refactored into the for loop above.
        return torch.stack(scatter_deltas_list).sum(dim=0), deltas_data # ORIGINAL
        # return torch.stack(scatter_deltas_list).sum(dim=0), scatter_deltas_list
