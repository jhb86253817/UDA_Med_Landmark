
class Config():
    def __init__(self):
        self.task = 'head'
        self.semi_iter = 5
        self.curriculum = [0.2, 0.4, 0.6, 0.8, 1]
        self.origin_size = (2400, 1935)
        self.input_size = (800, 640)
        self.phy_dist = 0.1
        self.batch_size = 10
        self.init_lr = 0.0002
        self.num_epochs = 720
        self.decay_steps = [480, 640]
        self.backbone = 'resnet50'
        self.pretrained = True
        self.criterion_cls = 'l2'
        self.criterion_reg = 'l1'
        self.cls_loss_weight = 100
        self.xy_loss_weight = 0.02
        self.domain_loss_weight = 0.01
        self.gt_sigma = 2
        self.num_lms = 19
        self.use_gpu = True
        self.gpu_id = 0
        self.tf_dim = 256
        self.tf_en_num = 0
        self.tf_de_num = 3
