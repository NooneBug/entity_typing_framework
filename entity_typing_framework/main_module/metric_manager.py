from torchmetrics import Precision, Recall

class MetricManager():

    def __init__(self, num_classes, device):
        self.micro_p = Precision(num_classes=num_classes, average='micro', mdmc_average='global').to(device=device)
        self.micro_r = Recall(num_classes=num_classes, average='micro', mdmc_average='global').to(device=device)
        
        self.macro_p_ex = Precision(num_classes=num_classes, average='samples', mdmc_average='global').to(device=device)
        self.macro_r_ex = Recall(num_classes=num_classes, average='samples', mdmc_average='global').to(device=device)
        
        self.macro_p_t = Precision(num_classes=num_classes, average='macro', mdmc_average='global').to(device=device)
        self.macro_r_t = Recall(num_classes=num_classes, average='macro', mdmc_average='global').to(device=device)

    def set_device(self, device):
        self.micro_p = self.micro_p.to(device=device)
        self.micro_r = self.micro_r.to(device=device)

        self.macro_p_ex = self.macro_p_ex.to(device=device)
        self.macro_r_ex = self.macro_r_ex.to(device=device)

        self.macro_p_t = self.macro_p_t.to(device=device)
        self.macro_r_t = self.macro_r_t.to(device=device)

    def update(self, pred, target):
        pred = pred.float()
        target = target.int()
        self.micro_p.update(preds=pred, target=target)
        self.micro_r.update(preds=pred, target=target)

        self.macro_p_ex.update(preds=pred, target=target)
        self.macro_r_ex.update(preds=pred, target=target)

        self.macro_p_t.update(preds=pred, target=target)
        self.macro_r_t.update(preds=pred, target=target)
    
    def compute(self):
        micro_p = self.micro_p.compute()
        micro_r = self.micro_r.compute()
        micro_f1 = self.compute_f1(micro_p, micro_r)

        macro_p_ex = self.macro_p_ex.compute()
        macro_r_ex = self.macro_r_ex.compute()
        macro_f1_ex = self.compute_f1(macro_p_ex, macro_r_ex)

        macro_p_t = self.macro_p_t.compute()
        macro_r_t = self.macro_r_t.compute()
        macro_f1_t = self.compute_f1(macro_p_t, macro_r_t)

        return self.compose_return(micro_p, micro_r, micro_f1, macro_p_ex, macro_r_ex, macro_f1_ex, macro_p_t, macro_r_t, macro_f1_t)

    def compose_return(self, micro_p, micro_r, micro_f1, macro_p_ex, macro_r_ex, macro_f1_ex, macro_p_t, macro_r_t, macro_f1_t):
        return {'micro/micro_precision' : micro_p,
                'micro/micro_recall' : micro_r,
                'micro/micro_f1' : micro_f1,
                'macro_example/macro_precision_example' :  macro_p_ex,
                'macro_example/macro_recall_example': macro_r_ex,
                'macro_example/macro_f1_example': macro_f1_ex,
                'macro_types/macro_precision_types' : macro_p_t,
                'macro_types/macro_recall_types': macro_r_t,
                'macro_types/macro_f1_types': macro_f1_t}

    def compute_f1(self, p, r):
        return (2 * p * r) / (p + r)