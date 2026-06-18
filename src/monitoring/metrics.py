import wandb


def log_metrics(metrics):

    wandb.log(metrics)