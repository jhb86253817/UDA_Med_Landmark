
class Config():
    def __init__(self):
        self.task = 'lung'
        self.semi_iter = 5
        self.curriculum = [0.2, 0.4, 0.6, 0.8, 1]
        self.origin_size = None
        self.input_size = (512, 512)
        self.phy_dist = 0.7
        self.batch_size = 10
        self.init_lr = 0.0002
        self.num_epochs = 720
        self.decay_steps = [480, 640]
        self.backbone = 'resnet50'
        self.pretrained = True
        self.criterion_cls = 'l2'
        self.criterion_reg = 'l1'
        self.cls_loss_weight = 15
        self.xy_loss_weight = 0.01
        self.domain_loss_weight = 0.005
        self.gt_sigma = 2
        self.num_lms = 94
        self.use_gpu = True
        self.gpu_id = 1
        self.tf_dim = 256
        self.tf_en_num = 0
        self.tf_de_num = 3
