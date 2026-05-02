import logging
import os
import random
# from threading import local
from model.backbones.vit_pytorch import deit_tiny_patch16_224_TransReID, part_attention_deit_small, part_attention_deit_tiny, part_attention_vit_base, part_attention_vit_base_p32, part_attention_vit_large, part_attention_vit_large_p14, part_attention_vit_large_p14_eva, part_attention_vit_small, vit_base_patch32_224_TransReID, vit_large_patch16_224_TransReID
import torch
import torch.nn as nn

from .backbones.resnet import BasicBlock, ResNet, Bottleneck
from .backbones import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID

# alter this to your pre-trained file name
lup_path_name = {
    'vit_base_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
    'vit_small_patch16_224_TransReID': 'vit_base_ics_cfs_lup.pth',
}

# alter this to your pre-trained file name
imagenet_path_name = {
    'vit_large_patch16_224_TransReID': 'jx_vit_large_p16_224-4ee7a4dc.pth',
    'vit_large_patch14_dinov2_TransReID': 'dinov2_vitl14_pretrain.pth',
    'vit_large_patch14_eva_TransReID': 'eva_large_patch14_196.pth',
    'vit_base_patch16_224_TransReID': 'jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_base_patch32_224_TransReID': 'jx_vit_base_patch32_224_in21k-8db57226.pth',
    'deit_base_patch16_224_TransReID': 'deit_base_distilled_patch16_224-df68dfff.pth',
    'vit_small_patch16_224_TransReID': 'vit_small_p16_224-15ec54c9.pth',
    'deit_small_patch16_224_TransReID': 'deit_small_distilled_patch16_224-649709d9.pth',
    'deit_tiny_patch16_224_TransReID': 'deit_tiny_distilled_patch16_224-b40b3cf7.pth'
}

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, model_name, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        
        # model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 2048
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
            model_path = os.path.join(model_path_base, \
                "resnet18-f37072fd.pth")
            print('using resnet18 as a backbone')
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet34-b627a593.pth")
            print('using resnet34 as a backbone')
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            model_path = os.path.join(model_path_base, \
                "resnet50-0676ba61.pth")
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
            model_path = os.path.join(model_path_base, \
                "resnet101-63fe2227.pth")
            print('using resnet101 as a backbone')
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            model_path = os.path.join(model_path_base, \
                "resnet152-394f9c45.pth")
            print('using resnet152 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # self.pool = nn.Linear(in_features=16*8, out_features=1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x) # B, C, h, w
        
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # global_feat = self.pool(x.flatten(2)).squeeze() # is GAP harming generalization?

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))


class build_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory):
        super(build_vit, self).__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
            self.in_planes = 192
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
            self.in_planes = 1024
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch14_dinov2_TransReID':
            self.in_planes = 1024
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch14_eva_TransReID':
            self.in_planes = 1024
        if self.pretrain_choice == 'imagenet':
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))
            
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.base(x) # B, N, C
        global_feat = x[:, 0] # cls token for global feature

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            if 'bottleneck' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))

'''
part attention vit
'''
class build_part_attention_vit(nn.Module):
    def __init__(self, num_classes, cfg, factory, pretrain_tag='imagenet'):
        super().__init__()
        self.cfg = cfg
        model_path_base = cfg.MODEL.PRETRAIN_PATH
        if pretrain_tag == 'lup':
            path = lup_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        else:
            path = imagenet_path_name[cfg.MODEL.TRANSFORMER_TYPE]
        self.model_path = os.path.join(model_path_base, path)
        self.pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: part token vit as a backbone')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE]\
            (img_size=cfg.INPUT.SIZE_TRAIN,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate= cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            pretrain_tag=pretrain_tag)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        elif cfg.MODEL.TRANSFORMER_TYPE == 'deit_tiny_patch16_224_TransReID':
            self.in_planes = 192
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch16_224_TransReID':
            self.in_planes = 1024
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch14_dinov2_TransReID':
            self.in_planes = 1024
        elif cfg.MODEL.TRANSFORMER_TYPE == 'vit_large_patch14_eva_TransReID':
            self.in_planes = 1024
        if self.pretrain_choice == 'imagenet':
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......from {}'.format(self.model_path))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # Deep-supervision heads for last N intermediate transformer blocks.
        # Each aux head has its own BN + Linear classifier. The processor
        # computes CE+Triplet on each (aux_score, aux_cls_feat) pair and sums
        # them with MODEL.AUX_LOSS_WEIGHT. Accessed only when MODEL.DEEP_SUP.
        self.deep_sup = cfg.MODEL.DEEP_SUP
        self.num_aux = cfg.MODEL.NUM_AUX_LAYERS
        if self.deep_sup:
            self.aux_bottlenecks = nn.ModuleList([
                nn.BatchNorm1d(self.in_planes) for _ in range(self.num_aux)
            ])
            self.aux_classifiers = nn.ModuleList([
                nn.Linear(self.in_planes, self.num_classes, bias=False) for _ in range(self.num_aux)
            ])
            for bn in self.aux_bottlenecks:
                bn.bias.requires_grad_(False)
                bn.apply(weights_init_kaiming)
            for cl in self.aux_classifiers:
                cl.apply(weights_init_classifier)
            print(f'deep supervision ON: {self.num_aux} aux heads on layers -2..-{self.num_aux+1}')

        # Camera-adversarial head (gradient reversal layer to make features
        # camera-invariant). Trained simultaneously with the ID classifier.
        # Activated by MODEL.CAM_ADV. Backbone gets NEGATIVE gradient from this
        # head, so it learns to produce features the camera classifier can't
        # distinguish. Targets the c004 cross-camera generalization gap.
        self.cam_adv = cfg.MODEL.CAM_ADV
        if self.cam_adv:
            self.num_cameras = cfg.MODEL.NUM_CAMERAS  # default 3 (c001-c003 in training)
            self.cam_adv_lambda = cfg.MODEL.CAM_ADV_LAMBDA
            self.cam_classifier = nn.Linear(self.in_planes, self.num_cameras, bias=False)
            self.cam_classifier.apply(weights_init_classifier)
            print(f'camera-adversarial ON: classifier over {self.num_cameras} cams, GRL lambda={self.cam_adv_lambda}')

    def forward(self, x, label=None, cam_label=None):
        layerwise_tokens = self.base(x) # B, N, C
        layerwise_cls_tokens = [t[:, 0] for t in layerwise_tokens] # cls token
        part_feat_list = layerwise_tokens[-1][:, 1: 4] # 3, 768

        layerwise_part_tokens = [[t[:, i] for i in range(1,4)] for t in layerwise_tokens] # 12 3 768
        feat = self.bottleneck(layerwise_cls_tokens[-1])

        if self.training:
            # ArcFace path: replace plain linear classifier with cos(theta+margin)
            # logits using normalized classifier weights. Requires `label` to know
            # which class entry to apply margin to. If label is missing, fall
            # back to plain classifier (preserves backward compatibility).
            if self.cfg.MODEL.ID_LOSS_TYPE == 'arcface' and label is not None:
                from loss.arcface_head import arcface_logits
                cls_score = arcface_logits(
                    feat, self.classifier.weight, label,
                    scale=self.cfg.MODEL.ARCFACE_S,
                    margin=self.cfg.MODEL.ARCFACE_M,
                )
            else:
                cls_score = self.classifier(feat)
            if self.deep_sup:
                aux_scores = []
                for i in range(self.num_aux):
                    aux_cls = layerwise_cls_tokens[-(i + 2)]  # -2, -3, ..., -(num_aux+1)
                    aux_feat = self.aux_bottlenecks[i](aux_cls)
                    aux_scores.append(self.aux_classifiers[i](aux_feat))
                return cls_score, layerwise_cls_tokens, layerwise_part_tokens, aux_scores
            # Camera-adversarial path: emit camera logits with reversed gradient.
            # Returns 4-tuple (cls_score, cls_tokens, part_tokens, cam_logits).
            if self.cam_adv:
                from utils.gradient_reversal import grad_reverse
                cam_logits = self.cam_classifier(grad_reverse(feat, self.cam_adv_lambda))
                return cls_score, layerwise_cls_tokens, layerwise_part_tokens, cam_logits
            return cls_score, layerwise_cls_tokens, layerwise_part_tokens
        else:
            return feat if self.neck_feat == 'after' else layerwise_cls_tokens[-1]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i: # drop classifier
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def compute_num_params(self):
        total = sum([param.nelement() for param in self.parameters()])
        logger = logging.getLogger('PAT.train')
        logger.info("Number of parameter: %.2fM" % (total/1e6))        

__factory_T_type = {
    'vit_large_patch16_224_TransReID': vit_large_patch16_224_TransReID,
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_base_patch32_224_TransReID': vit_base_patch32_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'deit_tiny_patch16_224_TransReID': deit_tiny_patch16_224_TransReID,
}

__factory_LAT_type = {
    'vit_large_patch16_224_TransReID': part_attention_vit_large,
    'vit_large_patch14_dinov2_TransReID': part_attention_vit_large_p14,
    'vit_large_patch14_eva_TransReID': part_attention_vit_large_p14_eva,
    'vit_base_patch16_224_TransReID': part_attention_vit_base,
    'vit_base_patch32_224_TransReID': part_attention_vit_base_p32,
    'deit_base_patch16_224_TransReID': part_attention_vit_base,
    'vit_small_patch16_224_TransReID': part_attention_vit_small,
    'deit_small_patch16_224_TransReID': part_attention_deit_small,
    'deit_tiny_patch16_224_TransReID': part_attention_deit_tiny,
}

class build_seresnet50(nn.Module):
    """SE-ResNet-50 (BoT-style). Wraps timm seresnet50 with BNNeck.

    Returns a 3-tuple matching PAT processor's expected shape so we can reuse
    the existing training loop. layerwise_cls_tokens[-1] is the pre-bottleneck
    feature used for triplet; layerwise_part_tokens are dummy for compatibility
    (PC_LOSS must be disabled in the YAML for SE-ResNet-50 since it has no
    explicit part tokens).
    """
    def __init__(self, num_classes, cfg):
        super().__init__()
        import timm
        self.base = timm.create_model('seresnet50', pretrained=True,
                                       num_classes=0, global_pool='avg')
        self.in_planes = 2048
        self.num_classes = num_classes
        self.cfg = cfg
        self.neck_feat = cfg.TEST.NECK_FEAT

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        print('===========building SE-ResNet-50 (BoT-style) ===========')
        print(f'  feature dim: {self.in_planes}, classes: {self.num_classes}')

    def forward(self, x, label=None, cam_label=None):
        feat_raw = self.base(x)              # (B, 2048) global-pooled feature
        feat = self.bottleneck(feat_raw)     # BNNeck

        if self.training:
            cls_score = self.classifier(feat)
            # Match PAT processor's 3-tuple shape; layerwise lists have only
            # the final feature since ResNet has no per-block CLS analog.
            layerwise_cls_tokens = [feat_raw]
            # Dummy 3 part-tokens (= same feature repeated) — never used since
            # PC_LOSS=False for this model.
            layerwise_part_tokens = [[feat_raw, feat_raw, feat_raw]]
            return cls_score, layerwise_cls_tokens, layerwise_part_tokens
        else:
            return feat if self.neck_feat == 'after' else feat_raw

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            k = i.replace('module.', '')
            if k in self.state_dict():
                self.state_dict()[k].copy_(param_dict[i])
        print('Loading trained model from {}'.format(trained_path))

    def compute_num_params(self):
        total = sum([p.nelement() for p in self.parameters()])
        logger = logging.getLogger('PAT.train')
        if logger.handlers:
            logger.info("Number of parameter: %.2fM" % (total/1e6))


def make_model(cfg, modelname, num_class, sd_flag=False, head_flag=False, camera_num=None, view_num=None):
    if modelname == 'vit':
        model = build_vit(num_class, cfg, __factory_T_type)
        print('===========building vit===========')
    elif modelname == 'part_attention_vit':
        model = build_part_attention_vit(num_class, cfg, __factory_LAT_type)
        print('===========building our part attention vit===========')
    elif modelname == 'seresnet50':
        model = build_seresnet50(num_class, cfg)
    else:
        model = Backbone(modelname, num_class, cfg)
        print('===========building ResNet===========')
    ### count params
    model.compute_num_params()
    return model