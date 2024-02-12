'''
CUDA_VISIBLE_DEVICES=3 python -u domainnet.py --cfg cfgs/cotta.yaml
'''
import logging

import torch
import torch.optim as optim
import torchvision.models as tmodels

import tent
import norm
import cotta

from load_domainnet import DomainNetLoader, get_domainnet126

from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)

model_path = {
    'clipart':'/home/yxue/model_fusion_dnn/ckpt_res50_domainnet126/checkpoint/ckpt_clipart__sgd_lr-s0.001_lr-w-1.0_bs32_seed42_source-[]_DomainNet126_resnet50-1.0x_SingleTraining-DomainNet126_lrd-[-2, -1]_wd-0.0005.pth',
    'painting':'/home/yxue/model_fusion_dnn/ckpt_res50_domainnet126/checkpoint/ckpt_painting__sgd_lr-s0.001_lr-w-1.0_bs32_seed42_source-[]_DomainNet126_resnet50-1.0x_SingleTraining-DomainNet126_lrd-[-2, -1]_wd-0.0005.pth',
    'real':'/home/yxue/model_fusion_dnn/ckpt_res50_domainnet126/checkpoint/ckpt_real__sgd_lr-s0.001_lr-w-1.0_bs32_seed42_source-[]_DomainNet126_resnet50-1.0x_SingleTraining-DomainNet126_lrd-[-2, -1]_wd-0.0005.pth',
    'sketch':'/home/yxue/model_fusion_dnn/ckpt_res50_domainnet126/checkpoint/ckpt_sketch__sgd_lr-s0.001_lr-w-1.0_bs32_seed42_source-[]_DomainNet126_resnet50-1.0x_SingleTraining-DomainNet126_lrd-[-2, -1]_wd-0.0005.pth',
}


def evaluate(description):
    load_cfg_fom_args(description)

    # configure model
    base_model = tmodels.resnet50(num_classes=126).cuda()
    base_model.load_state_dict(torch.load(model_path['real'])['net'])
    
    targets = ['painting', 'sketch']  # 'clipart', 'painting', 'real', 'sketch'
    
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)


    LEN_SET_DomainNet126 = {
        'clipart':18523,
        'painting':30042,
        'real':69622,
        'sketch':24147,
    }

    # _, _, test_ls = DomainNetLoader(
    #     dataset_path='/home/yxue/datasets/DomainNet',
    #     batch_size=cfg.TEST.BATCH_SIZE,
    #     num_workers=16,
    # ).get_source_dloaders(domain_ls=targets)
    
    test_ls = []
    for d in targets:
        test_loader = get_domainnet126('/home/yxue/datasets/DomainNet-126', d, cfg.TEST.BATCH_SIZE)
        test_ls.append(test_loader)

    for idx, test_loader in enumerate(test_ls):
        correct_num = 0
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            res = model(data)
            _, predicted = torch.max(res.data, 1)
            correct = predicted.eq(label.data).cpu().sum()
            correct_num += correct
        acc = correct_num / LEN_SET_DomainNet126[targets[idx]]
        print(f'{acc:.4}')
           


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    else:
        raise NotImplementedError

def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


if __name__ == '__main__':
    evaluate('"DomainNet evaluation.')
