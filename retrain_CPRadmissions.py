import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from EpiLearnSpatialTemporal.dataset import UniversalDataset
from EpiLearnSpatialTemporal import metrics
from EpiLearnSpatialTemporal.utils import generate_dataset
from EpiLearnSpatialTemporal.base import EinnModule

from EpiLearnSpatialTemporal.AGCRN import AGCRN
from EpiLearnSpatialTemporal.ColaGNN import ColaGNN
from EpiLearnSpatialTemporal.DCRNN import DCRNN
from EpiLearnSpatialTemporal.Dlinear import DlinearModel
from EpiLearnSpatialTemporal.EpiGNN import EpiGNN
from EpiLearnSpatialTemporal.EARTH import EARTH
from EpiLearnSpatialTemporal.GraphWaveNet import GraphWaveNet
from EpiLearnSpatialTemporal.MTGNN import MTGNN
from EpiLearnSpatialTemporal.STGCN import STGCN
from EpiLearnSpatialTemporal.GTS import GTS
from EpiLearnSpatialTemporal.StemGNN import StemGNN
from EpiLearnSpatialTemporal.STNorm import STNorm


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_splits(lookback=28, horizon=7, train_rate=0.6, val_rate=0.2, permute=False):
    data_df = pd.read_csv("rawData/processed/CPRadmissions.csv", index_col = 0)
    data_df.index = pd.to_datetime(data_df.index)
    raw_df = data_df.astype(np.float32).copy()
    raw_df = raw_df.clip(lower=0.0)
    pos_mask = raw_df > 0
    pos_vals = raw_df.where(pos_mask)
    std_s = pos_vals.std(axis=0, skipna=True).replace(0, 1.0).fillna(1.0)
    scaler = {
        "std": torch.as_tensor(std_s.values, dtype=torch.float32),
        "zero_preserve": True,
        "center": False,
    }
    data_df = raw_df.copy()
    nonzero = data_df > 0
    data_df[nonzero] = data_df[nonzero] / std_s
    data_df[~nonzero] = 0.0
    adj_df = pd.read_csv("rawData/processed/CPRadmissions_adj.csv", index_col = 0)

    dataset = UniversalDataset()
    data = np.expand_dims(data_df.values, axis=-1)
    dataset.x = torch.FloatTensor(data)
    dataset.y = torch.FloatTensor(data)[:, :, 0]
    dataset.graph = torch.FloatTensor(adj_df.to_numpy())

    dow = torch.as_tensor(data_df.index.dayofweek.values, dtype=torch.long)
    # woy = torch.as_tensor(data_df.index.isocalendar().week.values - 1, dtype=torch.long)
    dataset.states = torch.stack([dow], dim=-1)
    tid_s = {'dow': 7}

    train_dataset, val_dataset, test_dataset = dataset.ganerate_splits(
        train_rate=train_rate, val_rate=val_rate
    )

    train_input, train_target, _, train_states_future, train_adj = generate_dataset(
        X=train_dataset["features"],
        Y=train_dataset["target"],
        states=train_dataset["states"],
        dynamic_adj=train_dataset["dynamic_graph"],
        lookback_window_size=lookback,
        horizon=horizon,
        permute=permute,
    )
    val_input, val_target, _, val_states_future, val_adj = generate_dataset(
        X=val_dataset["features"],
        Y=val_dataset["target"],
        states=val_dataset["states"],
        dynamic_adj=val_dataset["dynamic_graph"],
        lookback_window_size=lookback,
        horizon=horizon,
        permute=permute,
    )
    test_input, test_target, _, test_states_future, test_adj = generate_dataset(
        X=test_dataset["features"],
        Y=test_dataset["target"],
        states=test_dataset["states"],
        dynamic_adj=test_dataset["dynamic_graph"],
        lookback_window_size=lookback,
        horizon=horizon,
        permute=permute,
    )

    splits = {
        "train": {
            "features": train_input,
            "targets": train_target,
            "states": train_states_future,
            "dynamic_graph": train_adj,
        },
        "val": {
            "features": val_input,
            "targets": val_target,
            "states": val_states_future,
            "dynamic_graph": val_adj,
        },
        "test": {
            "features": test_input,
            "targets": test_target,
            "states": test_states_future,
            "dynamic_graph": test_adj,
        },
    }
    return data_df, dataset.graph, splits, tid_s, train_dataset, scaler


def compute_dtw_matrix(train_dataset, dataset_name, cache_dir="."):
    try:
        from fastdtw import fastdtw
    except ImportError as exc:
        raise ImportError("fastdtw is required to compute the DTW matrix.") from exc
    from tqdm import tqdm

    cache_path = os.path.join(cache_dir, f"dtw_{dataset_name}.npy")
    if os.path.exists(cache_path):
        dtw_matrix = np.load(cache_path)
        print(f"Loaded DTW matrix from {cache_path}")
        return dtw_matrix

    num_nodes = train_dataset["features"].shape[1]
    data_mean = train_dataset["features"].reshape(train_dataset["features"].shape[0], num_nodes, 1)
    dtw_matrix = np.zeros((num_nodes, num_nodes))
    for i in tqdm(range(num_nodes)):
        for j in range(i, num_nodes):
            dtw_distance, _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
            dtw_matrix[i][j] = dtw_distance
    for i in range(num_nodes):
        for j in range(i):
            dtw_matrix[i][j] = dtw_matrix[j][i]

    np.save(cache_path, dtw_matrix)
    print(f"Saved DTW matrix to {cache_path}")
    return dtw_matrix

class RepeatLastBaseline:
    def fit(self, **kwargs):
        return self

    def train(self):
        return self

    def eval(self):
        return self

    def parameters(self):
        return []

    def predict(self, x_eval, **kwargs):
        return x_eval[:, -1:, :, 0]


class ARIMABaseline:
    def __init__(self, horizon):
        self.horizon = horizon

    def fit(self, **kwargs):
        return self

    def train(self):
        return self

    def eval(self):
        return self

    def parameters(self):
        return []

    def predict(self, x_eval, **kwargs):
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        import warnings

        x_np = x_eval.detach().cpu().numpy()
        n_samples, _, n_nodes, _ = x_np.shape
        preds = np.zeros((n_samples, 1, n_nodes), dtype=np.float32)

        for i in range(n_samples):
            for j in range(n_nodes):
                series = x_np[i, :, j, 0]
                fallback = float(series[-1])
                if np.allclose(series, series[0]):
                    preds[i, 0, j] = fallback
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        fit = ARIMA(series, order=(1, 0, 0)).fit()
                    preds[i, 0, j] = fit.forecast(steps=self.horizon)[-1]
                except Exception:
                    preds[i, 0, j] = fallback

        return torch.as_tensor(preds, dtype=x_eval.dtype, device=x_eval.device)


def build_model(name, lookback, horizon, num_nodes, adj, tid_s, use_future_ti, device, dtw_matrix=None):
    if name == "ARIMA":
        return ARIMABaseline(horizon=horizon)
    if name == "repeat_last":
        return RepeatLastBaseline()
    common = dict(
        num_timesteps_input=lookback,
        num_timesteps_output=1,
        adj_m=adj,
        num_nodes=num_nodes,
        num_features=1,
        device=device,
        use_future_ti=use_future_ti,
        tid_sizes=tid_s,
        emb_dim=4,
        ti_hidden=(8,),
    )
    if name == "AGCRN":
        return AGCRN(rnn_units=4, nlayers=2, embed_dim=8, cheb_k=2, **common)
    if name == "ColaGNN":
        return ColaGNN(nhid=16, n_layer=2, **common)
    if name == "DCRNN":
        return DCRNN(
            max_diffusion_step=2,
            filter_type="dual_random_walk",
            num_rnn_layers=2,
            rnn_units=32,
            dropout=0.1,
            **common,
        )
    if name == "EpiGNN":
        return EpiGNN(k=5, hidA=16, hidR=4, hidP=1, n_layer=2, dropout=0.2, **common)
    if name == "EARTH":
        if dtw_matrix is None:
            raise ValueError("EARTH requires dtw_matrix.")
        return EARTH(
            dtw_matrix=dtw_matrix,
            dropout=0.2,
            n_hidden=8,
            **common,
        )
    if name == "GraphWaveNet":
        return GraphWaveNet(
            residual_channels=4,
            dilation_channels=4,
            skip_channels=32,
            end_channels=64,
            kernel_size=2,
            blocks=4,
            nlayers=8,
            **common,
        )
    if name == "MTGNN":
        return MTGNN(
            gcn_depth=3,
            dropout=0.2,
            subgraph_size=3,
            node_dim=8,
            dilation_exponential=1,
            conv_channels=8,
            residual_channels=4,
            skip_channels=8,
            end_channels=32,
            layers=3,
            propalpha=0.05,
            tanhalpha=3,
            **common,
        )
    if name == "STGCN":
        return STGCN(nhids=32, **common)
    if name == "GTS":
        return GTS(rnn_units=16, max_diffusion_step=2, **common)
    if name == "StemGNN":
        return StemGNN(stack_cnt=2, multi_layer=4, dropout_rate=0.2, leaky_rate=0.2, **common)
    if name == "STNorm":
        return STNorm(channels=8, kernel_size=2, blocks=4, layers=2, **common)
    if name == "Dlinear":
        return DlinearModel(
            num_timesteps_input=lookback,
            num_timesteps_output=1,
            num_features=1,
            num_nodes=num_nodes,
            use_future_ti=use_future_ti,
            tid_sizes=tid_s,
            emb_dim=4,
            ti_hidden=(8,),
            device=device,
        )
    raise ValueError(f"Unknown model name: {name}")


def eval_metrics(pred, target):
    mse = metrics.get_MSE(pred, target)
    mae = metrics.get_MAE(pred, target)
    rmse = metrics.get_RMSE(pred, target)
    mse_filtered = metrics.get_MSE_filtered(pred, target)
    mae_filtered = metrics.get_MAE_filtered(pred, target)
    medse = metrics.get_medSE(pred, target)
    medae = metrics.get_medAE(pred, target)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mse_filtered": mse_filtered,
        "mae_filtered": mae_filtered,
        "medse": medse,
        "medae": medae,
    }


def run_experiment(
    model_name,
    splits,
    adj,
    tid_s,
    use_future_ti,
    epi_mode,
    use_einn,
    loss_name,
    horizon,
    device,
    dtw_matrix=None,
    epochs=100,
):
    model = build_model(
        model_name,
        lookback=splits["train"]["features"].shape[1],
        horizon=horizon,
        num_nodes=adj.shape[0],
        adj=adj,
        tid_s=tid_s,
        use_future_ti=use_future_ti,
        device=device,
        dtw_matrix=dtw_matrix,
    )

    model.fit(
        train_input=splits["train"]["features"],
        train_target=splits["train"]["targets"],
        train_states=splits["train"]["states"],
        train_graph=adj,
        train_dynamic_graph=splits["train"]["dynamic_graph"],
        val_input=splits["val"]["features"],
        val_target=splits["val"]["targets"],
        val_states=splits["val"]["states"],
        val_graph=adj,
        val_dynamic_graph=splits["val"]["dynamic_graph"],
        loss=loss_name,
        epochs=epochs,
        use_epi_reg=False if not epi_mode else 0.1,
        epi_mode=epi_mode,
    )

    if use_einn and epi_mode and model_name not in ("ARIMA", "repeat_last"):
        einn = EinnModule(
            num_nodes=adj.shape[0],
            horizon=horizon,
            in_features=splits["train"]["features"].shape[-1],
            epi_mode=epi_mode,
        ).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(einn.parameters()), lr=1e-3
        )
        model.train()
        einn.train()
        y_hat = model(
            splits["train"]["features"],
            adj,
            splits["train"]["states"],
            splits["train"]["dynamic_graph"],
        )
        L_base = F.mse_loss(y_hat, splits["train"]["targets"])
        L_ode, L_data, y_einn = einn.losses(
            splits["train"]["features"],
            splits["train"]["targets"],
            graph=adj,
            dynamic_graph=splits["train"]["dynamic_graph"],
        )
        L_align = F.mse_loss(y_hat, y_einn)
        loss = L_base + 0.1 * L_ode + 0.1 * L_data + 0.1 * L_align
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        preds = model.predict(
            splits["test"]["features"],
            graph=adj,
            states=splits["test"]["states"],
            dynamic_graph=splits["test"]["dynamic_graph"],
        )

    targets = splits["test"]["targets"]
    out = eval_metrics(preds, targets)
    if model_name in ("ARIMA", "repeat_last"):
        out.update({"crps": float("nan"), "wis": float("nan")})
        return out
    model._fit_conformal(
        splits["val"]["features"],
        splits["val"]["targets"],
        states=splits["val"]["states"],
        graph=adj,
        dynamic_graph=splits["val"]["dynamic_graph"],
    )
    crps_wis = model.compute_crps_wis(
        splits["test"]["features"],
        targets,
        quantile_levels=(0.5, 0.05, 0.95, 0.10, 0.90, 0.15, 0.85),
        alphas=(0.10, 0.20, 0.30),
        graph=adj,
        states=splits["test"]["states"],
        dynamic_graph=splits["test"]["dynamic_graph"],
        n_samples=100,
    )
    out.update(crps_wis)
    return out


def save_metrics(metrics_out, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    data = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics_out.items()}
    data["tag"] = tag
    path = os.path.join(out_dir, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(pd.Series(data).to_json())
    return data

def run_retraining(
    dataset_name,
    data_df,
    adj,
    tid_s,
    model_name="Dlinear",
    horizon=7,
    lookback=28,
    retrain_every=90,
    retrain_train_length=180,
    first_target=None,
    out_dir="outputs_retrain",
    device="cpu",
    dtw_matrix=None,
    epochs=50,
    use_future_ti=True,
    epi_mode=False,
    use_einn=False,
    loss_name="mse",
    tag_prefix="",
    retrain_schedule=None,
):
    """Walk-forward retraining demo with per-retrain plotting."""
    n_total = len(data_df)
    if first_target is None:
        first_target = lookback + horizon - 1

    values = torch.FloatTensor(np.expand_dims(data_df.values, axis=-1))
    targets_full = values[:, :, 0]
    dow = torch.as_tensor(data_df.index.dayofweek.values, dtype=torch.long)
    if "woy" in tid_s:
        woy = torch.as_tensor(data_df.index.isocalendar().week.values - 1, dtype=torch.long)
        states = torch.stack([woy], dim=-1)
    else:
        states = torch.stack([dow], dim=-1)

    full_X, full_y, _, full_states, full_adj = generate_dataset(
        X=values,
        Y=targets_full,
        states=states,
        dynamic_adj=None,
        lookback_window_size=lookback,
        horizon=horizon,
        permute=False,
    )
    if retrain_schedule is None:
        retrain_schedule = []
        for start_idx in range(first_target, n_total, retrain_every):
            train_end = start_idx
            train_start = max(0, train_end - retrain_train_length)
            target_indices = list(range(start_idx, min(start_idx + retrain_every, n_total)))
            retrain_schedule.append(
                {
                    "start_idx": start_idx,
                    "train_start": train_start,
                    "train_end": train_end,
                    "target_indices": target_indices,
                }
            )
    rows = []
    retrain_id = 0
    for schedule in retrain_schedule:
        train_start = schedule["train_start"]
        train_end = schedule["train_end"]
        target_indices = schedule["target_indices"]

        subset_values = values[train_start:train_end]
        subset_targets = targets_full[train_start:train_end]
        subset_states = states[train_start:train_end]

        all_input, all_target, _, all_states_future, all_adj = generate_dataset(
            X=subset_values,
            Y=subset_targets,
            states=subset_states,
            dynamic_adj=None,
            lookback_window_size=lookback,
            horizon=horizon,
            permute=False,
        )
        n_windows = all_input.shape[0]
        if n_windows < 2:
            continue

        n_train = max(1, int(0.8 * n_windows))
        if n_train >= n_windows:
            n_train = n_windows - 1

        train_input = all_input[:n_train]
        train_target = all_target[:n_train]
        train_states_future = None if all_states_future is None else all_states_future[:n_train]
        train_adj = None if all_adj is None else all_adj[:n_train]

        val_input = all_input[n_train:]
        val_target = all_target[n_train:]
        val_states_future = None if all_states_future is None else all_states_future[n_train:]
        val_adj = None if all_adj is None else all_adj[n_train:]


        model = build_model(
            model_name,
            lookback=lookback,
            horizon=1,
            num_nodes=adj.shape[0],
            adj=adj,
            tid_s=tid_s,
            use_future_ti=use_future_ti,
            device=device,
            dtw_matrix=dtw_matrix,
        )

        model.fit(
            train_input=train_input,
            train_target=train_target,
            train_states=train_states_future,
            train_graph=adj,
            train_dynamic_graph=train_adj,
            val_input=val_input,
            val_target=val_target,
            val_states=val_states_future,
            val_graph=adj,
            val_dynamic_graph=val_adj,
            loss=loss_name,
            epochs=epochs,
            use_epi_reg=False if not epi_mode else 0.1,
            epi_mode=epi_mode,
        )

        if use_einn and epi_mode and model_name not in ("ARIMA", "repeat_last"):
            einn = EinnModule(
                num_nodes=adj.shape[0],
                horizon=horizon,
                in_features=train_input.shape[-1],
                epi_mode=epi_mode,
            ).to(device)
            optimizer = torch.optim.Adam(list(model.parameters()) + list(einn.parameters()), lr=1e-3)
            model.train()
            einn.train()
            y_hat = model(train_input, adj, train_states_future, train_adj)
            L_base = F.mse_loss(y_hat, train_target)
            L_ode, L_data, y_einn = einn.losses(
                train_input,
                train_target,
                graph=adj,
                dynamic_graph=train_adj,
            )
            L_align = F.mse_loss(y_hat, y_einn)
            total_loss = L_base + 0.1 * L_ode + 0.1 * L_data + 0.1 * L_align
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        offset = lookback + horizon - 1
        sample_ids = [t - offset for t in target_indices if 0 <= (t - offset) < full_X.shape[0]]
        valid_targets = [t for t in target_indices if 0 <= (t - offset) < full_X.shape[0]]
        if len(sample_ids) == 0:
            continue

        x_eval = full_X[sample_ids]
        y_true = full_y[sample_ids]
        states_eval = None if full_states is None else full_states[sample_ids]
        adj_eval = None if full_adj is None else full_adj[sample_ids]

        with torch.no_grad():
            y_pred = model.predict(
                x_eval,
                graph=adj,
                states=states_eval,
                dynamic_graph=adj_eval,
            )
        y_pred = y_pred.reshape(y_true.shape)

        for local_i, t_idx in enumerate(valid_targets):
            for state_idx, state_name in enumerate(data_df.columns):
                rows.append(
                    {
                        "retrain_id": retrain_id,
                        "timestamp": data_df.index[t_idx],
                        "state_idx": state_idx,
                        "state": str(state_name),
                        "train_start": data_df.index[train_start],
                        "train_end": data_df.index[train_end - 1],
                        "eval_start": data_df.index[target_indices[0]],
                        "eval_end": data_df.index[target_indices[-1]],
                        "y_true": y_true[local_i, 0, state_idx].item(),
                        "y_pred": y_pred[local_i, 0, state_idx].item(),
                    }
                )
        retrain_id += 1

    retrain_df = pd.DataFrame(rows)
    if retrain_df.empty:
        return retrain_df

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"retrain_{dataset_name}_{tag_prefix}.csv")
    retrain_df.to_csv(csv_path, index=False)
    return retrain_df


def plot_retraining_state_from_csv(csv_path, state=None, state_idx=None, out_path=None):
    """Read retraining CSV and plot one specific state across retrain windows."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows in {csv_path}")

    if state is None and state_idx is None:
        state = str(df["state"].iloc[0])

    if state is not None:
        sub_df = df[df["state"].astype(str) == str(state)].copy()
        label = f"state={state}"
    else:
        sub_df = df[df["state_idx"] == int(state_idx)].copy()
        label = f"state_idx={state_idx}"

    if sub_df.empty:
        raise ValueError(f"No rows found for {label} in {csv_path}")

    sub_df["timestamp"] = pd.to_datetime(sub_df["timestamp"])
    sub_df = sub_df.sort_values(["retrain_id", "timestamp"])

    plt.figure(figsize=(11, 4))
    for rid, grp in sub_df.groupby("retrain_id"):
        grp = grp.sort_values("timestamp")
        plt.plot(grp["timestamp"], grp["y_pred"], alpha=0.5, label=f"pred_r{rid}")

    truth = sub_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    plt.plot(truth["timestamp"], truth["y_true"], color="black", linewidth=1.5, label="y_true")
    plt.title(f"Rolling retraining predictions ({label})")
    plt.xlabel("timestamp")
    plt.ylabel("target")
    plt.legend(ncol=4, fontsize=7)
    plt.tight_layout()

    if out_path is None:
        suffix = str(state) if state is not None else f"idx_{state_idx}"
        out_path = csv_path.replace(".csv", f"_plot_{suffix}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

def main():
    dataset_name="CPRadmissions"
    fix_seed(42)
    device = "cpu"
    data_df, adj, splits, tid_s, train_dataset, scaler = build_splits()
    adj = adj.type(torch.float)
    dtw_matrix = compute_dtw_matrix(train_dataset, dataset_name=dataset_name)
    out_dir = f"retrain0301_{dataset_name}"
    # results = []

    model_names = [
        "MTGNN",
        "STGCN",
        "StemGNN",
        "STNorm",
        "EARTH",
        "GraphWaveNet",
    ]
    epi_modes = [False, "sir_incidence", "ngm"]
    loss_names = ["mse", "mse_filtered"]

    for horizon in [10]: 
        data_df, adj, splits, tid_s, train_dataset, scaler = build_splits(lookback=28, horizon=horizon, train_rate=0.6, val_rate=0.2)
        dtw_matrix = compute_dtw_matrix(train_dataset, dataset_name=dataset_name)
        first_target = 28 + horizon - 1
        retrain_schedule = []
        for start_idx in range(first_target, len(data_df), 90):
            retrain_schedule.append({
                "start_idx": start_idx,
                "train_start": max(0, start_idx - 180),
                "train_end": start_idx,
                "target_indices": list(range(start_idx, min(start_idx + 90, len(data_df)))),
            })
        for model_name in model_names:
            for epi_mode in epi_modes:
                for loss_name in loss_names:
                    use_filtering = loss_name == "mse_filtered"
                    for use_einn in (False, True):
                        if use_einn and not epi_mode:
                            continue
                        for use_future_ti in (True, False):
                            tag = (
                                f"{model_name}|horizon={horizon}|epi={epi_mode}|einn={use_einn}|filter={use_filtering}|ti={use_future_ti}"
                            )
                            retrain_tag = tag.replace("|", "__")
                            run_retraining(
                                dataset_name=dataset_name,
                                data_df=data_df,
                                adj=adj,
                                tid_s=tid_s,
                                model_name=model_name,
                                horizon=horizon,
                                lookback=28,
                                retrain_every=90,
                                retrain_train_length=180,
                                out_dir=out_dir,
                                device=device,
                                dtw_matrix=dtw_matrix if model_name == "EARTH" else None,
                                epochs=50,
                                use_future_ti=use_future_ti,
                                epi_mode=epi_mode,
                                use_einn=use_einn,
                                loss_name=loss_name,
                                tag_prefix=retrain_tag,
                                retrain_schedule=retrain_schedule,
                            )
                            csv_path = os.path.join(
                                out_dir,
                                f"retrain_{dataset_name}_{retrain_tag}.csv",
                            )
                            plot_retraining_state_from_csv(
                                csv_path,
                                state='PA',
                            )
                            # metrics_out = run_experiment(
                            #     model_name=model_name,
                            #     splits=splits,
                            #     adj=adj,
                            #     tid_s=tid_s,
                            #     use_future_ti=use_future_ti,
                            #     epi_mode=epi_mode,
                            #     use_einn=use_einn,
                            #     loss_name=loss_name,
                            #     horizon=splits["train"]["targets"].shape[1],
                            #     device=device,
                            #     dtw_matrix=dtw_matrix if model_name == "EARTH" else None,
                            # )
                            # results.append(save_metrics(metrics_out, out_dir, tag))
                            # df = pd.DataFrame(results)
                            # df.to_csv(os.path.join(out_dir, f"metrics_{dataset_name}.csv"), index=False)

if __name__ == "__main__":
    main()