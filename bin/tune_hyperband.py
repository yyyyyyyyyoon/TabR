import os
import sys
import glob
import json

project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

os.environ["PROJECT_DIR"] = project_dir
sys.path.append(project_dir)

import optuna
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data import preprocess_data
from tabr import Model, evaluate
import lib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate_model(csv_path, d_main, encoder_n_blocks, context_size, predictor_n_blocks, trial=None):
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    splits = preprocess_data(csv_path)

    if dataset_name not in splits:
        raise ValueError(f"{dataset_name} not found in preprocessed splits.")

    X_all = splits[dataset_name]["X_train"]
    y_all = splits[dataset_name]["y_train"]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        X_tr_tensor = torch.from_numpy(X_tr).float().to(device)
        X_te_tensor = torch.from_numpy(X_te).float().to(device)
        y_tr_tensor = torch.from_numpy(y_tr.to_numpy()).long().to(device)
        y_te_tensor = torch.from_numpy(y_te.to_numpy()).long().to(device)

        model = Model(
            n_num_features=X_tr.shape[1],
            n_bin_features=1,
            cat_cardinalities=[],
            n_classes=2,
            num_embeddings=None,
            d_main=d_main,
            d_multiplier=2.0,
            encoder_n_blocks=encoder_n_blocks,
            predictor_n_blocks=predictor_n_blocks,
            mixer_normalization='auto',
            context_dropout=0.1,
            dropout0=0.2,
            dropout1='dropout0',
            normalization='LayerNorm',
            activation='ReLU',
            memory_efficient=False,
            candidate_encoding_batch_size=None,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()

        n_epochs = 50
        for epoch in range(n_epochs):
            model.train()
            for batch_idx in lib.make_random_batches(len(y_tr_tensor), 32, device):
                batch_x = X_tr_tensor[batch_idx]
                batch_y = y_tr_tensor[batch_idx]
                x_ = {'num': batch_x}
                candidate_x_ = {'num': X_tr_tensor}
                candidate_y = y_tr_tensor

                optimizer.zero_grad()
                output = model(x_, batch_y, candidate_x_, candidate_y, context_size=context_size, is_train=True)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()

            val_metrics = evaluate(model, X_te_tensor, y_te_tensor, X_tr_tensor, y_tr_tensor, context_size=context_size)

            if trial is not None:
                adjusted_score = val_metrics["PD"] - 0.5 * val_metrics["PF"]
                trial.report(adjusted_score, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        metrics = evaluate(model, X_te_tensor, y_te_tensor, X_tr_tensor, y_tr_tensor, context_size=context_size)
        all_metrics.append(metrics)

    return pd.DataFrame(all_metrics).mean().to_dict()

def objective(trial, csv_path):
    d_main = trial.suggest_categorical("d_main", [64, 128, 256, 365])
    encoder_n_blocks = trial.suggest_int("encoder_n_blocks", 0, 3)
    context_size = trial.suggest_categorical("context_size", [32, 64, 96, 128, 160])
    predictor_n_blocks = trial.suggest_int("predictor_n_blocks", 1, 3)

    score = train_and_evaluate_model(
        csv_path=csv_path,
        d_main=d_main,
        encoder_n_blocks=encoder_n_blocks,
        context_size=context_size,
        predictor_n_blocks=predictor_n_blocks,
        trial=trial
    )

    trial.set_user_attr("PD", score["PD"])
    trial.set_user_attr("FIR", score["FIR"])
    trial.set_user_attr("PF", score["PF"])
    trial.set_user_attr("Balance", score["Balance"])
    return score["PD"] - 0.5 * score["PF"]


if __name__ == '__main__':
    data_root = "data"
    dataset_paths = glob.glob(os.path.join(data_root, "*", "*.csv"))
    os.makedirs("results", exist_ok=True)

    for csv_path in dataset_paths:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"Tuning on: {csv_path}")

        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
        study.optimize(lambda trial: objective(trial, csv_path), n_trials=100, timeout=3600, n_jobs=4)  # 30분 제한

        best = study.best_trial
        print(f" Best adjusted score (PD - 0.5 * PF): {best.value:.4f}")
        print(f"    PD:     {best.user_attrs['PD']:.4f}")
        print(f"    PF:     {best.user_attrs['PF']:.4f}")
        print(f"    FIR:    {best.user_attrs['FIR']:.4f}")
        print(f"    Balance: {best.user_attrs['Balance']:.4f}")
        print(f"    Params: {best.params}")

        with open(f"results/{dataset_name}_10fold_50epoch.json", "w") as f:
            json.dump({
                "dataset": dataset_name,
                "adjusted_score": best.value,
                "PD": best.user_attrs["PD"],
                "PF": best.user_attrs["PF"],
                "FIR": best.user_attrs["FIR"],
                "Balance": best.user_attrs["Balance"],
                "params": best.params,
            }, f, indent=2)
