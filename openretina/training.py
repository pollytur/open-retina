import datetime
import os
from functools import partial

try:
    import wandb
except ImportError:
    print("wandb not installed, not logging to wandb")

import numpy as np
import torch
from tqdm.auto import tqdm

from . import measures, metrics
from .cyclers import LongCycler
from .early_stopping import early_stopping
from .tracking import MultipleObjectiveTracker
from .utils.misc import set_seed
from sklearn.cluster import KMeans
from torch.nn import KLDivLoss


def standard_early_stop_trainer(
    model: torch.nn.Module,
    dataloaders,
    seed: int,
    scale_loss: bool = True,  # trainer args
    loss_function: str = "PoissonLoss3d",
    stop_function: str = "corr_stop",
    loss_accum_batch_n=None,
    device: str = "cuda",
    verbose: bool = True,
    interval: int = 1,
    patience: int = 5,
    epoch: int = 0,
    lr_init: float = 0.005,  # early stopping args
    max_iter: int = 100,
    maximize: bool = True,
    tolerance: float = 1e-6,
    restore_best: bool = True,
    lr_decay_steps: int = 3,
    lr_decay_factor: float = 0.3,
    min_lr: float = 0.0001,  # lr scheduler args
    detach_core: bool = False,
    wandb_logger=None,
    cb=None,
    include_kldivergence=True,
    cluster_number=10,
    alpha=1.0,
    dec_starting_epoch=5,
    kmeans_init=20,
    base_multiplier=4e3,
    subsamples=2000,
    use_diag_cov=True,
    # learn_alpha=False,
    exponent=2,
    # load_pretrain=False,
    # load_adlognorm=True,
    # pretrained_epoch=30,
    **kwargs,
):
    
    def get_multiplier(epoch, base_multiplier=4e3):
        """Multiplier to scale KL loss in same order of magnitude as main loss
        To avoid hard peek aat starting epoch we include a warm-up phase s.t. the loss can increase slower
        """
        if epoch < dec_starting_epoch:
            return 0
        else:
            return base_multiplier

    def soft_assignments(encoded_features, cluster_centers, alpha=alpha):
        """
        Compute soft assingments q_ij as described in DEC paper (1)
        q_ij = (1+ ||z_i - \mu_j||^2/a)^(-(a+1)/2) / (sum_j'((1+ ||z_i - \mu_j'||^2/a)^(-(a+1)/2)))
        """
        norm_squared = torch.sum(
            (encoded_features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, 2
        )
        assignments = 1.0 / (1.0 + (norm_squared / alpha))
        assignments = assignments ** ((alpha + 1) / 2)
        return assignments / torch.sum(assignments, dim=1, keepdim=True)

    def target_distribution(batch: torch.Tensor, exponent=exponent) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        p_ij = (q_ij^2/f_j) / sum_j'(q_ij'^2/f_j')  f_j =sum_i(q_ij)

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch**exponent) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


    def soft_assignments_mult(encoded_features, cluster_centers, sigma, alpha, p=1):
        sigma_inv = 1.0 / sigma  # (K, D)
        diff = encoded_features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)  # (N, K, D)
        norm_sigma = torch.sum(diff * sigma_inv * diff, dim=2)  # (N, K)
        det = torch.sum(torch.log(sigma), dim=1)  # log(det) since sigma is diagonal
        log_gamma_top = torch.lgamma((alpha + p) / 2)
        log_gamma_bottom = torch.lgamma(alpha / 2)
        # Log-density formula for multivariate Student-t
        log_pdf = (
            log_gamma_top
            - log_gamma_bottom
            - 0.5 * det
            - (p / 2) * torch.log(alpha * torch.pi)
            - ((alpha + p) / 2) * torch.log(1 + (norm_sigma / alpha))
        )
        #log_pdf_max = torch.max(log_pdf, dim=1, keepdim=True)[0]  # Get max per row

        log_assignments = log_pdf - torch.logsumexp(log_pdf, dim=1, keepdim=True)
        return torch.exp(log_assignments)  # Convert log-assignments to probabilities

    def EM_t_mult(features, resp, cluster_centers, sigma, alpha, d=1):
        sigma_inv = 1.0 / sigma  # (K,)
        diff = features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)
        norm_sigma = torch.sum((diff**2 * sigma_inv), 2)
        u = ((alpha + d) / (alpha + norm_sigma)).detach()  # ccalculate U shape(N,K)
        
        """ M step """
        numerator = torch.matmul(features, resp * u).T.detach()
        denominator = torch.sum(resp * u, dim=0, keepdim=True).T.detach()
        cluster_centers = numerator / denominator

        weighted_sq_diff = resp.unsqueeze(2) * u.unsqueeze(2) * (diff**2)  # (N, K, D)
        numerator = weighted_sq_diff.sum(dim=0)  # (K,D)
        denominator = torch.sum(resp, dim=0, keepdim=True)  # (K,)
        sigma = (numerator / denominator.T).detach()

        sigma = torch.clamp(sigma, min=1e-4, max=1e4)

        return cluster_centers, sigma

    def EM_t_1D(features, resp, cluster_centers, taus, alpha, d=1):
        norm_squared = torch.sum(
            (features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, dim=2
        )
        u = (alpha + d) / (
            alpha + norm_squared * (taus ** (-1))
        )  # ccalculate U shape(N,K)
        print("u", u)

        """ M step """
        numerator = torch.matmul(features, resp * u).T.detach()
        # print(numerator.shape)
        denominator = torch.sum(resp * u, dim=0, keepdim=True).T.detach()
        print("denom cc", denominator)
        cluster_centers = numerator / denominator

        weighted_sums = torch.sum(resp * u * norm_squared, dim=0)
        taus = (weighted_sums / torch.sum(resp, dim=0, keepdim=True)).detach()
        print("Tau", taus)
        return cluster_centers, taus

    def soft_assignments_1D(encoded_features, cluster_centers, tau, alpha=1):
        norm_squared = torch.sum(
            (encoded_features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, 2
        )
        assignments = 1.0 / (1.0 + (norm_squared / (alpha * tau)))
        assignments = (assignments ** ((alpha + 1) / 2)) / (tau**1 / 2)
        return assignments / torch.sum(assignments, dim=1, keepdim=True)
    
    # Defines objective function; criterion is resolved to the loss_function that is passed as input
    def full_objective(model, data_key, inputs, targets, detach_core):
        regularizers = int(not detach_core) * model.core.regularizer() + model.readout.regularizer(data_key)
        if scale_loss:
            m = len(trainloaders[data_key].dataset)
            k = inputs.shape[0]
            loss_scale = np.sqrt(m / k)
        else:
            loss_scale = 1.0

        predictions = model(inputs.to(device), data_key, detach_core=detach_core)
        loss_criterion = criterion(predictions, targets.to(device))
        res = loss_scale * loss_criterion + regularizers
        return res

    trainloaders = dataloaders["train"]
    valloaders = dataloaders.get("validation", dataloaders["val"] if "val" in dataloaders.keys() else None)
    testloaders = dataloaders["test"]

    # Model training
    model.to(device)
    set_seed(seed)
    model.train()

    kldiv_criterion = KLDivLoss(
        size_average=False
    )  # losses are summed for each minibatch

    criterion = getattr(measures, loss_function)()
    stop_closure = partial(getattr(metrics, stop_function), model, valloaders, device=device)

    n_iterations = len(LongCycler(trainloaders))

    alpha = torch.tensor(alpha, device=device, requires_grad=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(trainloaders.keys()) if loss_accum_batch_n is None else loss_accum_batch_n
    )  # will be equal to number of sessions (dict keys) if not specified

    # define some trackers
    tracker_dict = dict(
        val_correlation=partial(metrics.corr_stop3d, model, valloaders, device=device),
        val_poisson_loss=partial(metrics.poisson_stop3d, model, valloaders, device=device),
        val_MSE_loss=partial(metrics.MSE_stop3d, model, valloaders, device=device),
    )

    if hasattr(model, "tracked_values"):
        tracker_dict.update(model.tracked_values)

    tracker = MultipleObjectiveTracker(**tracker_dict)
    
    kldiv_list = []
    print("Alpha: ", alpha)

    # train over epochs
    for epoch, val_obj in tqdm(
        early_stopping(
            model,
            stop_closure,
            interval=interval,
            patience=patience,
            start=epoch,
            max_iter=max_iter,
            maximize_objective=maximize,
            tolerance=tolerance,
            restore_best=restore_best,
            tracker=tracker,
            scheduler=scheduler,
            lr_decay_steps=lr_decay_steps,
        ),
        desc="Epochs",
        total=max_iter,
        position=0,
        leave=True,
    ):
        if include_kldivergence and epoch == dec_starting_epoch:
            cluster_centers_list = []
            kmeans = KMeans(
                n_clusters=cluster_number, n_init=kmeans_init, random_state=seed
            )
            feature_list = []
            # form initial cluster centres
            with torch.no_grad():
                for k, readout in model.readout.items():
                    features = readout.features.cpu().detach().squeeze().T.numpy()
                    feature_list.append(np.array(features))

                features = np.vstack(feature_list)
                predicted = kmeans.fit_predict(features)

            cluster_centers = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float, device=device
            )
            if use_diag_cov:
                p = features.shape[1]
                sigma = torch.zeros((cluster_number, p), device=device)
                for k in range(cluster_number):
                    cluster_points = torch.from_numpy(features[predicted == k]).to(
                        device
                    )
                    print(f"Points for cluster {k}: {cluster_points.shape[0]}")
                    if len(cluster_points) > 0:
                        sigma[k] = (
                            torch.var(cluster_points, dim=0, unbiased=True) + 1e-6
                        )

            else:
                sigma = torch.zeros(cluster_number, device=device)
                for k in range(cluster_number):
                    cluster_points = torch.from_numpy(features[predicted == k]).to(
                        device
                    )
                    if len(cluster_points) > 0:
                        sigma[k] = torch.mean(
                            torch.sum((cluster_points - cluster_centers[k]) ** 2, 1)
                        )
                sigma = sigma.unsqueeze(0)

        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        epoch_loss = 0
        # epoch_loss_main = 0
        # epoch_loss_reg = 0
        epoch_loss_kldiv = 0
        epoch_loss_kldiv_without_scaling = 0
        # epoch_kldiv_loss_regularizer = 0
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(trainloaders)),
            total=n_iterations,
            desc=f"Epoch {epoch}",
            position=1,
            leave=True,
            disable=not verbose,
        ):
            clean_data_key = clean_session_key(data_key)
            loss = full_objective(model, clean_data_key, *data, detach_core)  # type: ignore
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                if include_kldivergence and epoch >= dec_starting_epoch:
                    kldiv_loss = torch.zeros(1).to(device)
                    feature_list = []
                    for k, readout in model.readout.items():
                        features = readout.features.squeeze()
                        feature_list.append(features)
                    feature_list = torch.cat(feature_list, dim=1)
                    if use_diag_cov:
                        q = soft_assignments_mult(
                            feature_list, cluster_centers, sigma, alpha, p
                        )
                    else:
                        q = soft_assignments_1D(
                            feature_list, cluster_centers, sigma, alpha
                        )
                    target = target_distribution(q, exponent)
                    target = target.clamp(min=1e-10)
                    q = q.clamp(min=1e-10)

                    kldiv_loss = get_multiplier(epoch, base_multiplier) * (
                        kldiv_criterion(q.log(), target)
                    )

                    # To avoid underflow issues when computing this quantity, this loss expects the argument input in the log-space.
                    # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                    kldiv_loss.backward()
                    epoch_loss_kldiv += kldiv_loss.detach()
                    epoch_loss_kldiv_without_scaling += (
                        kldiv_loss.detach() / get_multiplier(epoch, base_multiplier)
                    )
                    epoch_loss += kldiv_loss.detach()
                    with torch.no_grad():
                        cluster_centers_list.append(cluster_centers.cpu().detach())
                        kldiv_list.append(
                            kldiv_loss.cpu() / get_multiplier(epoch, base_multiplier)
                        )

                    if use_diag_cov:
                        cluster_centers, sigma = EM_t_mult(
                            feature_list, q, cluster_centers, sigma, alpha, p
                        )
                    else:
                        cluster_centers, sigma = EM_t_1D(
                            feature_list, q, cluster_centers, sigma, alpha
                        )
                optimizer.step()
                optimizer.zero_grad()
            if np.isnan(loss.item()):
                raise ValueError(f"Loss is NaN on batch {batch_no} from {data_key}, stopping training.")
        if wandb_logger is not None:
            tracker_info = tracker.asdict(make_copy=True)
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "Batch": batch_no,
                    "val_corr": tracker_info["val_correlation"][-1],
                    "val_poisson_loss": tracker_info["val_poisson_loss"][-1],
                    "val_MSE_loss": tracker_info["val_MSE_loss"][-1],

                    "Epoch Train loss Kullback-Leibler-divergence": epoch_loss_kldiv,
                    "Epoch Train loss KL without scaling main": epoch_loss_kldiv_without_scaling,
                }
            )

    # Model evaluation
    model.eval()
    if include_kldivergence:
        soft_assignments_list = []
        for k, readout in model.readout.items():
            features = readout.features.detach().squeeze()
            soft_assignments_list.append(
                soft_assignments_mult(features, cluster_centers, sigma, alpha, p)
            )
        predicted = torch.cat(soft_assignments_list).max(1)[1]
        # append final cluster_centers
        cluster_centers_list.append(cluster_centers.cpu().detach().numpy())
        cluster_centers_np = np.array(cluster_centers_list)
        kldiv_list_np = np.array(kldiv_list)
        print("Alpha: ", alpha)

    tracker.finalize()

    # Compute avg validation and test correlation
    avg_val_corr = metrics.corr_stop3d(model, valloaders, avg=True, device=device)
    avg_test_corr = metrics.corr_stop3d(model, testloaders, avg=True, device=device)

    # return the whole tracker output as a dict
    output = tracker.asdict()
    if include_kldivergence:
        output['cluster_centers_np'] = cluster_centers_np
        output['predicted'] = predicted

    if wandb_logger is not None:
        wandb.finish()
    return avg_test_corr, avg_val_corr, output, model.state_dict()


def save_checkpoint(model, optimizer, epoch, loss, save_folder: str, model_name: str) -> None:
    if not os.path.exists(save_folder):
        # only create the lower level directory if it does not exist
        os.mkdir(save_folder)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join(save_folder, f"{model_name}_{date}_checkpoint.pt"),
    )


def save_model(model: torch.nn.Module, save_folder: str, model_name: str) -> None:
    if not os.path.exists(save_folder):
        # only create the lower level directory if it does not exist
        os.mkdir(save_folder)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    torch.save(model.state_dict(), os.path.join(save_folder, f"{model_name}_{date}_model_weights.pt"))
    torch.save(model, os.path.join(save_folder, f"{model_name}_{date}_model.pt"))


def clean_session_key(session_key: str) -> str:
    # Ignore this function when only training on the chirp or movingbar by uncommenting the following line
    # return session_key
    if "_chirp" in session_key:
        session_key = session_key.split("_chirp")[0]
    if "_mb" in session_key:
        session_key = session_key.split("_mb")[0]
    return session_key
