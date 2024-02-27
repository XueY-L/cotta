'''
CUDA_VISIBLE_DEVICES=1 python -u imagenetc_forget.py --cfg cfgs/cotta_11domains.yaml
'''
import copy
import logging

import torch
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from robustbench.data import load_imagenetc
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
    source = 'jpeg_compression'
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()  # 最前面是normalize
    base_model.load_state_dict(torch.load(f'/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'{source}\']_[1].pt')['model'])
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

    dataset = ImageFolder(f'/home/yxue/datasets/ImageNet-C/{source}/1', transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()]))
    source_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=16)

    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"
    for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            # try:
            #     if i_x == 0:
            #         model.reset()
            #         logger.info("resetting model")
            #     else:
            #         logger.warning("not resetting model")
            # except:
            #     logger.warning("not resetting model")
            x_test, y_test = load_imagenetc(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            print(f'{acc:.4}')
            
            num_cor = 0
            loader = copy.deepcopy(source_loader)
            temp_model = copy.deepcopy(model)
            for data, label in loader:
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    rst = temp_model(data)
                _, predicted = torch.max(rst.data, 1)
                correct = predicted.eq(label.data).cpu().sum()
                num_cor += correct
            
            print(f'{num_cor/len(loader):.4}')


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
    evaluate('"Imagenet-C evaluation.')
