"""
staged_optuna_search.py
-----------------------
Three-stage optimisation pipeline for the Deep-Fake detector:

1.  macro_sweep()  – coarse, discrete architecture search
2.  bohb_tuning()  – BOHB hyper-parameter tuning per candidate architecture
3.  zoom_in()      – local refinement around the incumbent

Author: <you>
"""

from __future__ import annotations
import copy
import os
import optuna
from typing import List, Dict, Tuple, Any
from constants import*
from data_methods import get_dataloader
from train_methods import train_model
from optimization import (
    AVDNet,  # model class
    EarlyStopping, setup_optimizer, save_best_model_callback, set_global_seed  # utilities
)

# ---------------------------------------------------------------------
# Helper: build the model given a (possibly partial) param dict
# ---------------------------------------------------------------------
def build_model(params: Dict) -> AVDNet:
    """Instantiate DeepFakeDetector from a param dictionary."""
    defaults = dict(             # sensible defaults for *untuned* knobs
        backbone="vgg",
        freeze_cnn=True,
        freeze_cnn_layers=8,
        freeze_wav2vec=True,
        freeze_feature_extractor=True,
        freeze_encoder_layers=0,
        d_model=256,
        nhead=8,
        num_layers=2,
        dense_hidden_dims=[256, 128],
        dropout=0.3,
        aug_prob = 0.2
    )
    cfg = {**defaults, **params}
    model = AVDNet(
        backbone=cfg["backbone"],
        freeze_cnn=cfg["freeze_cnn"],
        freeze_cnn_layers=cfg["freeze_cnn_layers"],
        freeze_wav2vec=cfg["freeze_wav2vec"],
        freeze_feature_extractor=cfg["freeze_feature_extractor"],
        freeze_encoder_layers=cfg["freeze_encoder_layers"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dense_hidden_dims=cfg["dense_hidden_dims"],
    ).to(DEVICE)

    # broadcast dropout hyper-parameter over every dropout module
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout,
                          torch.nn.Dropout2d,
                          torch.nn.Dropout3d)):
            m.p = cfg["dropout"]
    return model


# ---------------------------------------------------------------------
# 1)  MACRO ARCHITECTURE SWEEP  (discrete + cheap)
# ---------------------------------------------------------------------
def macro_sweep(n_trials: int = 30,
                study_name: str = "macro_sweep",
                storage: str | None = None,
                dataset_root: str = "/home/hp4ran/DeepFakeProject",
                subset_fraction: float = 0.3,
                top_k = 2
                ) -> List[Dict]:
    """
    Returns a list with the *k* best architecture parameter dictionaries.
    """
    # ––– define the *coarse* search space –––
    def objective_macro(trial: optuna.Trial) -> float:
        # architecture knobs only
        head_dim = trial.suggest_int("head_dim", 80, 172, step=16)  # or choose an appropriate range
        dense_initial_dim = trial.suggest_int("dense_initial_dim", 256, 1600, step=64)
        transformer_nhead = trial.suggest_int("transformer_nhead", 8, 24)
        dense_layers = trial.suggest_int("dense_layers", 2, 5)  # total number of dense layers in classifier

        params = {
            "freeze_cnn_layers": trial.suggest_int("freeze_cnn_layers", 0, 12, step=4),
            "freeze_encoder_layers": trial.suggest_int("freeze_encoder_layers", 0, 8, step=2),
            "num_layers": trial.suggest_int("num_layers", 1, 4, step=1),
            "nhead": transformer_nhead,
            "d_model": head_dim * transformer_nhead,

        }

        dense_hidden_dims = []
        current_dim = dense_initial_dim
        for _ in range(dense_layers - 1):
            next_dim = current_dim // 2
            dense_hidden_dims.append(next_dim)
            current_dim = next_dim

        params["dense_hidden_dims"] = dense_hidden_dims

        # dataloaders (very small subset, 2–3 epochs)
        train_dl = get_dataloader("Train",     DATASET_FOLDER,
                                  batch_size=16, num_workers=2,
                                  fraction=subset_fraction,
                                  data_aug=DEFAULT_AUG_PROB)
        val_dl   = get_dataloader("Validation", DATASET_FOLDER,
                                  batch_size=16, num_workers=2,
                                  fraction=subset_fraction)

        model = build_model(params)
        criterion = torch.nn.BCEWithLogitsLoss()
        optim     = setup_optimizer(model, learning_rate=1e-6, weight_decay=0)

        early_stop = EarlyStopping(patience=2)
        best, last, f1 = train_model(float("inf"), criterion,
                                    early_stop, model, optim, None,
                                    train_dl, trial, val_dl)

        # Store the best validation loss for the trial
        trial.set_user_attr("best_val_loss", best)

        return best   # we minimise validation loss

    study = optuna.create_study(direction="minimize",
                                study_name=study_name,
                                sampler=optuna.samplers.RandomSampler(),  # Sobol would work too
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage=storage,
                                load_if_exists=True)
    study.optimize(objective_macro, n_trials=n_trials, show_progress_bar=True,)

    # pick the top-k candidates for the next stage
    top_archs = []
    valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for t in sorted(valid_trials, key=lambda tr: tr.value)[:top_k]:
        top_archs.append(copy.deepcopy(t.params))
    return top_archs


# ---------------------------------------------------------------------
# 2)  BOHB HYPER-PARAMETER TUNING  (per candidate architecture)
# ---------------------------------------------------------------------
def bohb_tuning(arch_candidates: List[Dict],
                n_trials: int = 300,
                dataset_root: str = "/home/hp4ran/DeepFakeProject",
                study_prefix: str = "bohb",
                subset_fraction: float = 0.5,
                top_k : int = 4,
                storage = None
                ) -> List[Dict]:
    """Returns (best_param_dict, best_trial)."""

    candidate_trials: List[Tuple[Tuple[float, float], Dict]] = []

    # identical HP space for every surviving architecture
    def objective_fine(trial: optuna.Trial, fixed_arch: Dict) -> (float, float):
        params = fixed_arch.copy()
        params.update(
            learning_rate = trial.suggest_float("learning_rate", 1e-7, 3e-4, log=True),
            batch_size = trial.suggest_categorical("batch_size", [8, 16]),
            dropout = trial.suggest_float("dropout", 0.0, 0.8),
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
            aug_prob = trial.suggest_float("aug_prob", 0.0, 0.8)
        )

        # data
        bs = params["batch_size"]
        aug_prob = params["aug_prob"]

        # Loading the data
        train_dl = get_dataloader("Train", DATASET_FOLDER, batch_size=bs, num_workers=2,
                                      fraction=subset_fraction, data_aug=aug_prob)
        val_dl = get_dataloader("Validation", DATASET_FOLDER, batch_size=bs, num_workers=2,
                                    fraction=subset_fraction)

        model = build_model(params)
        criterion = torch.nn.BCEWithLogitsLoss()
        optim = setup_optimizer(model, params["learning_rate"],
                                    params["weight_decay"])
        early = EarlyStopping(patience=PATIENCE)

        best, last, f1 = train_model(float("inf"), criterion, early,
                                    model, optim, None, train_dl, trial, val_dl)

        trial.set_user_attr("full_params", params)  # keep full dict
        return last, f1

    # ––– run one study *per* architecture –––
    for i, arch in enumerate(arch_candidates):
        study = optuna.create_study(
            directions=["minimize", "maximize"],
            study_name=f"{study_prefix}_{i}",
            sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
            pruner=optuna.pruners.HyperbandPruner(min_resource=5, reduction_factor=3),
            storage=storage,
            load_if_exists=True
        )

        study.optimize(lambda tr: objective_fine(tr, arch),
                       n_trials=n_trials, show_progress_bar=True)

        for tr in study.trials:
            if tr.values is not None:  # completed
                candidate_trials.append((tuple(tr.values), tr.user_attrs["full_params"]))

    # sort & keep top-k
    candidate_trials.sort(key=lambda x: (-x[0][0], x[0][1]))
    top_k_params = [p for _, p in candidate_trials[:top_k]]
    return top_k_params


# ---------------------------------------------------------------------
# 3)  ZOOM-IN REFINEMENT  (re-centre sampling)
# ---------------------------------------------------------------------
def zoom_in(top_params: List[Dict],
            n_trials: int = 80,
            dataset_root: str = "/home/hp4ran/DeepFakeProject",
            subset_fraction :float = 1.0,
            storage = None) -> List[Dict]:
    """
    Narrows each *continuous* range around the incumbent and re-optimises.
    For simplicity we only resample learning-rate & dropout, keeping others frozen.
    """

    numeric_ranges: Dict[str, Tuple[Any, Any]] = {}
    categorical_choices: Dict[str, List[Any]] = {}
    params_to_update = ["learning_rate", "weight_decay", "dropout", "aug_prob"]
    for d in top_params:
        for k, v in [(k, v) for (k,v) in d.items() if k in params_to_update]:
            if isinstance(v, (int, float)):
                numeric_ranges[k] = v, v
            else:  # str / bool etc.
                categorical_choices.setdefault(k, []).append(v)

    # keep uniques for categorical
    categorical_choices = {k: sorted(set(vs))
                           for k, vs in categorical_choices.items()}

    study = optuna.create_study(direction="maximize",
                                study_name=f"FINAL_OPTIMIZATION",
                                pruner=optuna.pruners.MedianPruner(),
                                storage=storage,
                                load_if_exists=True)
    final_params = []
    for incumbent in top_params:
        def objective_zoom(trial: optuna.Trial) -> float:
            params = copy.deepcopy(incumbent)
            for k in params.keys():
                if k in numeric_ranges:
                    lo, hi = numeric_ranges[k]
                    # widen by ±10 % for exploration
                    span = hi - lo if isinstance(lo, float) else hi - lo
                    lo_ = lo - 0.1 * span
                    hi_ = hi + 0.1 * span

                    if isinstance(lo, float):
                        params[k] = trial.suggest_float(k, max(lo_, 1e-7), hi_, log=(k in {"learning_rate", "weight_decay"}))
                    else:  # int
                        params[k] = trial.suggest_int(k, lo, hi)
                elif k in categorical_choices and len(categorical_choices[k]) > 1:
                    params[k] = trial.suggest_categorical(k, categorical_choices[k])
                # else: keep incumbent value (fixed)

            # Loading the data
            train_dl = get_dataloader("Train", DATASET_FOLDER, batch_size=params["batch_size"], num_workers=2,
                                      fraction=subset_fraction, data_aug=params["aug_prob"])
            val_dl = get_dataloader("Validation", DATASET_FOLDER, batch_size=params["batch_size"], num_workers=2,
                                    fraction=subset_fraction)

            model = build_model(params)
            criterion = torch.nn.BCEWithLogitsLoss()
            optim = setup_optimizer(model, params["learning_rate"], params["weight_decay"])
            early = EarlyStopping(patience=PATIENCE)

            best, last, f1 = train_model(float("inf"), criterion, early,
                                   model, optim, None, train_dl, trial, val_dl)

            trial.set_user_attr("full_params", params)  # keep full dict
            return f1


        study.optimize(objective_zoom, n_trials=n_trials, show_progress_bar=True)

        final_params.append({**incumbent, **study.best_trial.params})


    return final_params


# ---------------------------------------------------------------------
# 4)  MAIN DRIVER
# ---------------------------------------------------------------------
def main() -> None:
    # Directories and paths
    os.makedirs("checkpoints", exist_ok=True)
    BEST_MODEL_PATH = "checkpoints/best_model.pth"
    BEST_PARAMS_PATH = "checkpoints/best_params.json"
    STUDY_DB_PATH = "sqlite:///checkpoints/3_stage_optimization.db"
    if not LOAD_TRAINING:
        STUDY_DB_PATH = None

    # set random seed
    set_global_seed(SEED)

    project_root = DATASET_FOLDER
    print("—— Stage 1 : Macro architecture sweep ——")
    top_archs = macro_sweep(n_trials=30,
                            dataset_root=project_root,
                            subset_fraction=0.30,
                            storage = STUDY_DB_PATH,
                            top_k= 4)
    print(top_archs)

    print("\n—— Stage 2 : BOHB hyper-parameter tuning ——")
    top_params = bohb_tuning(top_archs,
                             n_trials=30,
                             dataset_root=project_root,
                             subset_fraction=0.30,
                             storage = STUDY_DB_PATH,
                             top_k= 5) # 100

    print("\n—— Stage 3 : Local zoom-in refinement ——")
    final_params = zoom_in(top_params,
                           n_trials=2,
                           dataset_root=project_root,
                           subset_fraction=1,
                           storage = STUDY_DB_PATH)

    print("\n==========  Finished  ==========")
    print("Best params after three stages:")
    for k, v in final_params.items():
        print(f"{k}: {v}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
