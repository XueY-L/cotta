'''
CUDA_VISIBLE_DEVICES=2 python cifar100c.py --cfg cfgs/cotta_11domains.yaml
'''
import copy
import logging
import time

import torch
import torch.optim as optim

from robustbench.data import load_cifar100c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
import norm
import cotta

from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    source = 'snow'
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    base_model.load_state_dict(torch.load(f'/home/yuanxue/SAR/ckpt/cifar100/corruptions/ckpt_cifar100_[\'{source}\']_[1].pt')['model'])
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
    
    x_test_source, y_test_source = load_cifar100c(cfg.CORRUPTION.NUM_EX, 1, cfg.DATA_DIR, False, [source])
    x_test_source, y_test_source = x_test_source.cuda(), y_test_source.cuda()
    
    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"
    res, res_forget = [], []
    t1 = time.time()
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # continual adaptation for all corruption 
            # if i_c == 0:
            #     try:
            #         model.reset()
            #         logger.info("resetting model")
            #     except:
            #         logger.warning("not resetting model")
            # else:
            #     logger.warning("not resetting model")
            x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            # logger.info(f"accuracy % [{corruption_type}{severity}]: {acc:.4} \t error % [{corruption_type}{severity}]: {err:.4}")
            res.append(round(acc, 4))

            # 遗忘
            temp_model = copy.deepcopy(model)
            acc = accuracy(temp_model, x_test_source, y_test_source, cfg.TEST.BATCH_SIZE)
            res_forget.append(round(acc, 4))
            print(res, res_forget)
    t2 = time.time()
    logger.info(f'Time: {t2-t1}s')
        

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
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


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
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"CIFAR-100-C evaluation.')
