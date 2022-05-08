import math
import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import Callable, Tuple
from torch.utils.data import DataLoader
from data2vec.utils.misc import log, save_model
import os
import time
from tqdm import tqdm


def compute_var(y):
    y = y.view(-1, y.size(-1))
    return torch.sqrt(y.var(dim=0) + 1e-6).mean()


def variance_loss(z_a, z_b):
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-6).mean()
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-6).mean()
    var_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
    return var_loss


def train_single_batch(net: nn.Module, data: torch.Tensor, mask: torch.Tensor, optimizer: optim.Optimizer,
                       criterion: Callable, device: torch.device) -> Tuple[float, int]:
    """Performs a single training step.

    Args:
        net (nn.Module): Model instance.
        data (torch.Tensor): Data tensor, of shape (batch_size, dim_1, ... , dim_N).
        targets (torch.Tensor): Target tensor, of shape (batch_size).
        optimizer (optim.Optimizer): Optimizer instance.
        criterion (Callable): Loss function.
        device (torch.device): Device.

    Returns:
        float: Loss scalar.
        int: Number of correct preds.
    """

    data = data.to(device)

    optimizer.zero_grad()
    predictions, targets = net(data, data, mask)
    scale = math.sqrt(predictions.size(dim=-1))
    loss = criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale) + \
           25*variance_loss(predictions.float(), targets.float())
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        target_var = compute_var(targets.float())
        prediction_var = compute_var(predictions.float())
    return loss.item(), target_var.item(), prediction_var.item()


@torch.no_grad()
def evaluate(net: nn.Module, mask_generator, criterion: Callable, dataloader: DataLoader, device: torch.device) -> \
        Tuple[float, float]:
    """Performs inference.

    Args:
        net (nn.Module): Model instance.
        criterion (Callable): Loss function.
        dataloader (DataLoader): Test or validation loader.
        device (torch.device): Device.

    Returns:
        accuracy (float): Accuracy.
        float: Loss scalar.
    """

    net.eval()
    correct = 0
    running_loss = 0.0
    running_target_var = 0.0
    running_prediction_var = 0.0

    for data in tqdm(dataloader):
        data = data.to(device)
        batch_size = data.size(dim=0)
        audio_length = data.size(dim=-1)
        mask = mask_generator(shape=(batch_size, audio_length)).to("cuda")
        mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device), mask], dim=1).bool()

        predictions, targets = net(data, data, mask)
        scale = math.sqrt(predictions.size(dim=-1))
        loss = criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)
        target_var = compute_var(targets.float())
        prediction_var = compute_var(predictions.float())
        running_loss += loss.item()
        running_target_var += target_var.item()
        running_prediction_var += prediction_var.item()

    net.train()
    avg_loss = running_loss / len(dataloader.dataset)
    avg_target_var = running_target_var / len(dataloader)
    avg_prediction_var = running_prediction_var / len(dataloader)
    return avg_loss, avg_target_var, avg_prediction_var


def train(net: nn.Module, mask_generator, optimizer: optim.Optimizer, criterion: Callable, trainloader: DataLoader,
          valloader: DataLoader, schedulers: dict, config: dict) -> None:
    """Trains model.

    Args:
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer): Optimizer instance.
        criterion (Callable): Loss function.
        trainloader (DataLoader): Training data loader.
        valloader (DataLoader): Validation data loader.
        schedulers (dict): Dict containing schedulers.
        config (dict): Config dict.
    """

    step = 0
    best_avg_loss = 0.0
    n_batches = len(trainloader)
    device = config["hparams"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

    ############################
    # start training
    ############################
    net.train()

    for epoch in range(config["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss = 0.0
        running_target_var = 0.
        running_prediction_var = 0.
        for batch_index, data, in enumerate(trainloader):
            batch_size = data.size(dim=0)
            audio_length = data.size(dim=-1)
            if schedulers["warmup"] is not None and epoch < config["hparams"]["scheduler"]["n_warmup"]:
                schedulers["warmup"].step()

            elif schedulers["scheduler"] is not None:
                schedulers["scheduler"].step()

            ####################
            # optimization step
            ####################
            mask = mask_generator(shape=(batch_size, audio_length)).to("cuda")
            mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device), mask], dim=1).bool()

            loss, target_var, prediction_var = train_single_batch(net, data, mask, optimizer, criterion, device)
            net.ema_step()
            running_loss += loss
            running_target_var += target_var
            running_prediction_var += prediction_var

            if not step % config["exp"]["log_freq"]:
                log_dict = {"epoch": epoch, "loss": loss, "lr": optimizer.param_groups[0]["lr"],
                            "target_var": target_var, "prediction_var": prediction_var}
                log(log_dict, step, config)

            step += 1

        #######################
        # epoch complete
        #######################

        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0,
                    "avg_train_target_var": running_target_var / n_batches,
                    "avg_train_prediction_var": running_prediction_var / n_batches,
                    "avg_loss_per_ep": running_loss / len(trainloader.dataset)}
        log(log_dict, step, config)

        if not epoch % config["exp"]["val_freq"]:
            avg_val_loss, avg_val_target_var, avg_val_prediction_var = evaluate(net, mask_generator, criterion,
                                                                                valloader, device)
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss,
                        "avg_val_target_var": avg_val_target_var, "avg_val_prediction_var": avg_val_prediction_var}
            log(log_dict, step, config)

            # save best val ckpt
            if avg_val_loss < best_avg_loss or epoch == config["exp"]["val_freq"]:
                best_avg_loss = avg_val_loss
                save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
                save_model(epoch, avg_val_loss, save_path, net, optimizer, log_file)
                save_path = os.path.join(config["exp"]["save_dir"], "best_encoder.pth")
                save_model(epoch, avg_val_loss, save_path, net.encoder, optimizer, log_file)

                ###########################
    # training complete
    ###########################

    avg_val_loss, avg_val_target_var, avg_val_prediction_var = evaluate(net, mask_generator, criterion, valloader,
                                                                        device)
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss,
                "avg_val_target_var": avg_val_target_var, "avg_val_prediction_var": avg_val_prediction_var}

    log(log_dict, step, config)

    # save final ckpt
    save_path = os.path.join(config["exp"]["save_dir"], "last.pth")
    save_model(epoch, avg_val_loss, save_path, net, optimizer, log_file)
    save_path = os.path.join(config["exp"]["save_dir"], "last_encoder.pth")
    save_model(epoch, avg_val_loss, save_path, net.encoder, optimizer, log_file)
