from argparse import ArgumentParser
import datetime
from data2vec.config_parser import get_config

from data2vec.utils.loss import LabelSmoothingLoss
from data2vec.utils.opt import get_optimizer
from data2vec.utils.scheduler import WarmUpLR, get_scheduler
from data2vec.utils.trainer import train, evaluate
from data2vec.utils.dataset import get_train_val_test_split, get_loader
from data2vec.utils.misc import seed_everything, count_params, get_model, calc_step, log
from models.Data2Vec import Data2Vec
from models.KWT2 import kwt_from_name, KWT
from data2vec.masking import AudioMaskingGenerator

import torch
from torch import nn, optim
import numpy as np
import wandb

import os
import yaml
import random
import time


def training_pipeline(config):
    """Initiates and executes all the steps involved with model training.

    Args:
        config (dict) - Dict containing various settings for the training run.
    """

    config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    
    ######################################
    # save hyperparameters for current run
    ######################################

    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)
    
    #####################################
    # initialize training items
    #####################################

    # data
    with open(config["train_list_file"], "r") as f:
        train_list = f.read().rstrip().split("\n")
    
    with open(config["val_list_file"], "r") as f:
        val_list = f.read().rstrip().split("\n")

    trainloader = get_loader(train_list, config, train=True)
    valloader = get_loader(val_list, config, train=False)

    # mask generator
    mask_generator = AudioMaskingGenerator(mask_prob=config["hparams"]["model"]["mask_prob"],
                                                 mask_length=config["hparams"]["model"]["mask_length"],
                                                 attention_mask=None,
                                                 min_masks=config["hparams"]["model"]["min_masks"])

    # model
    if config["hparams"]["model"]["name"] is not None:
        encoder = kwt_from_name(config["hparams"]["model"]["name"])
    else:
        encoder = KWT(**config["hparams"]["model"])
    data2vec = Data2Vec(encoder=encoder,
                        modality=config["modality"],
                        model_embed_dim=config["hparams"]["model"]["dim"],
                        ema_decay=config["hparams"]["model"]["ema_decay"],
                        ema_end_decay=config["hparams"]["model"]["ema_end_decay"],
                        ema_anneal_end_step=config["hparams"]["model"]["ema_anneal_end_step"],
                        average_top_k_layers=config["hparams"]["model"]["average_top_k_layers"],
                        normalize_targets=config["hparams"]["model"]["normalize_targets"])
    model = data2vec.to(config["hparams"]["device"])
    print(f"Created model with {count_params(model)} parameters.")

    # loss
    criterion = nn.SmoothL1Loss(reduction="none", beta=config["hparams"]["loss_beta"])
    
    # optimizer
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=config["hparams"]["optimizer"]["opt_kwargs"]["lr"],
                           betas=config["hparams"]["optimizer"]["opt_kwargs"]["betas"],
                           eps=config["hparams"]["optimizer"]["opt_kwargs"]["eps"],
                           weight_decay=config["hparams"]["optimizer"]["opt_kwargs"]["weight_decay"])

    # lr scheduler
    epochs = config["hparams"]["n_epochs"]
    steps_per_epoch = len(trainloader)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["hparams"]["optimizer"]["opt_kwargs"]["lr"],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        anneal_strategy="cos")
    schedulers = {"scheduler": lr_scheduler,
                  "warmup": 0}


    #####################################
    # Training Run
    #####################################

    print("Initiating training.")
    train(model, mask_generator, optimizer, criterion, trainloader, valloader, schedulers, config)

    #####################################
    # Final Test
    #####################################

    with open(config["test_list_file"], "r") as f:
        test_list = f.read().rstrip().split("\n")

    testloader = get_loader(test_list, config, train=False)
    final_step = calc_step(config["hparams"]["n_epochs"] + 1, len(trainloader), len(trainloader) - 1)
   

    # evaluating the final state (last.pth)
    test_loss, test_target_var, test_prediction_var = evaluate(model, mask_generator, criterion, testloader, config["hparams"]["device"])
    log_dict = {
        "test_loss_last": test_loss,
        "test_target_var_last": test_target_var,
        "test_prediction_var_last": test_prediction_var,
    }
    log(log_dict, final_step, config)

    # evaluating the best validation state (best.pth)
    ckpt = torch.load(os.path.join(config["exp"]["save_dir"], "best.pth"))
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded.")

    test_loss, test_target_var, test_prediction_var = evaluate(model, mask_generator, criterion, testloader, config["hparams"]["device"])
    log_dict = {
        "test_loss_best": test_loss,
        "test_target_var_best": test_target_var,
        "test_prediction_var_best": test_prediction_var,
    }
    log(log_dict, final_step, config)


def main(args):
    config = get_config(args.conf)
    seed_everything(config["hparams"]["seed"])


    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()
        
        elif os.environ.get("WANDB_API_KEY", False):
            print(f"Found API key from env variable.")
        
        else:
            wandb.login()
        
        
        with wandb.init(project=config["exp"]["proj_name"], 
                        name=(config["exp"]["exp_name"]), 
                        config=config["hparams"]):
            training_pipeline(config)
    
    else:
        training_pipeline(config)



if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    args = parser.parse_args()

    main(args)