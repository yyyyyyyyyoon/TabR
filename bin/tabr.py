import pandas as pd
import os

if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import math
from dataclasses import dataclass
from typing import Literal, Optional, Union
from sklearn.model_selection import KFold

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from torch import Tensor
import data
import lib
from data import preprocess_data
import torch
from sklearn.metrics import confusion_matrix

from math import sqrt
from typing import Dict, Any

KWArgs = Dict[str, Any]
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

@dataclass(frozen=True)
class Config:
    seed: int
    data: Dict[str, numpy.ndarray]
    model: KWArgs  # Model
    context_size: int
    optimizer: KWArgs  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]


class Model(nn.Module):
    def __init__(
        self,
        *,
        #
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        #
        num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['auto']],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[float, Literal['dropout0']],
        normalization: str,
        activation: str,
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()

        # normalization 클래스 가져오기
        Normalization = getattr(nn, normalization)
        # activation 클래스 가져오기
        Activation = getattr(nn, activation)

        if dropout1 == 'dropout0':
            dropout1 = dropout0

        self.one_hot_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )
        self.num_embeddings = (
            None
            if num_embeddings is None
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )

        # >>> E
        n_bin_features = 0
        cat_cardinalities = []
        d_in = (
            n_num_features
            + n_bin_features
            + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, 2), # num_classes=2
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x_num = x_.get('num')
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')
        del x_

        x = []
        if x_num is None:
            assert self.num_embeddings is None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1)
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def forward(
        self,
        x_: dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        # >>>
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in delu.iter_batches(
                            candidate_x_, self.candidate_encoding_batch_size
                        )
                    ]
                )
            )
        x, k = self._encode(x_)
        if is_train:
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == 'cuda'
                    else faiss.IndexFlatL2(d_main)
                )
            self.search_index.reset()
            self.search_index.add(candidate_k)  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k, context_size + (1 if is_train else 0)
            )
            if is_train:
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            context_k = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass(frozen=True)
class Config:
    seed: int
    data: Dict[str, numpy.ndarray]
    model: KWArgs
    context_size: int
    optimizer: KWArgs
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]

context_size_default = 32

def evaluate(model, X_eval_tensor, y_eval_tensor, candidate_X_tensor, candidate_y_tensor, context_size=context_size_default, eval_batch_size=32):
    model.eval()
    preds = []
    with torch.inference_mode():
        while eval_batch_size:
            try:
                for i in range(0, len(X_eval_tensor), eval_batch_size):
                    batch_X = X_eval_tensor[i:i + eval_batch_size]
                    x_ = {'num': batch_X}
                    candidate_x_ = {'num': candidate_X_tensor}   # 후보 데이터 (예시로 학습 데이터 사용)
                    candidate_y = candidate_y_tensor
                    output = model(
                        x_=x_,
                        y=None,
                        candidate_x_=candidate_x_,
                        candidate_y=candidate_y,
                        context_size=context_size,
                        is_train=False,
                    )
                    preds.append(output.cpu())
            except RuntimeError as err:
                if "out of memory" not in str(err).lower():
                    raise
                eval_batch_size //= 2
            else:
                break

        preds = torch.cat(preds)
        _, y_pred = torch.max(preds, dim=1)
        y_true_np = y_eval_tensor.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        # confusion matrix, f1은 FIR 계산에 필요
        cm = confusion_matrix(y_true_np, y_pred_np)
        TP, FP, FN, TN = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
        FI = (TP + FP) / (TP + FP + TN + FN)
        PD = TP / (TP + FN) if (TP + FN) > 0 else 0
        PF = FP / (FP + TN) if (FP + TN) > 0 else 0
        FIR = (PD - FI) / PD if PD > 0 else 0
        Balance = 1 - (sqrt((0 - PF) ** 2 + (1 - PD) ** 2) / sqrt(2))

        return {'PD': PD, 'PF': PF, 'FIR': FIR, 'Balance': Balance}

def main() -> None:
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        print("Error: No CSV file path provided.")
        sys.exit(1)

    splits = preprocess_data(csv_path)
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    # 전처리된 데이터 가져오기
    if dataset_name not in splits:
        print(f"Error: Dataset '{dataset_name}' not found in splits.")
        sys.exit(1)

    X_all = splits[dataset_name]["X_train"]
    y_all = numpy.array(splits[dataset_name]["y_train"], dtype=int)

    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"\n[Fold {fold + 1}/{n_splits}]")

        # Fold별 데이터 나누기
        X_tr, X_te =  X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        # Tensor로 변환
        X_tr_tensor = torch.from_numpy(X_tr).float().to(device)
        X_te_tensor = torch.from_numpy(X_te).float().to(device)
        y_tr_tensor = torch.from_numpy(y_tr).long().to(device)
        y_te_tensor = torch.from_numpy(y_te).long().to(device)

        # model 생성
        model = Model(
            n_num_features=X_tr.shape[1],
            n_bin_features=1,
            cat_cardinalities=[],
            n_classes=2,
            num_embeddings=None,  # 또는 필요한 딕셔너리 제공
            d_main= 256,
            d_multiplier=2.0,
            encoder_n_blocks=3,
            predictor_n_blocks=2,
            mixer_normalization='auto',
            context_dropout=0.1,
            dropout0=0.2,
            dropout1='dropout0',  # 또는 float 값 제공
            normalization='LayerNorm',
            activation='ReLU',
            memory_efficient=False,
            candidate_encoding_batch_size=None,
        ).to(device)

        # 모델 훈련
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(50):
            model.train()
            epoch_loss = 0
            for batch_idx in lib.make_random_batches(len(y_tr_tensor), 32, device):
                batch_x =  X_tr_tensor[batch_idx]
                batch_y = y_tr_tensor[batch_idx]

                x_ = {'num': batch_x}
                candidate_x_ = {'num': X_tr_tensor}
                candidate_y = y_tr_tensor

                optimizer.zero_grad()
                output = model(x_, batch_y, candidate_x_, candidate_y, context_size=context_size_default, is_train=True)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(lib.make_random_batches(len(y_tr_tensor), 32, device))
            val_metrics = evaluate(model, X_te_tensor, y_te_tensor, X_tr_tensor, y_tr_tensor, context_size=context_size_default)

            # 에폭별 출력
            print(f"[Epoch {epoch + 1:02d}] Train Loss: {avg_loss:.4f} | "
                  f"PD: { val_metrics['PD']:.4f}, PF: {val_metrics['PF']:.4f}, "
                  f"FIR: {val_metrics['FIR']:.4f}, Balance: {val_metrics['Balance']:.4f}")

        eval_metrics = evaluate(model, X_te_tensor, y_te_tensor, X_tr_tensor, y_tr_tensor, context_size=context_size_default)
        all_metrics.append(eval_metrics)

    print("\n=== [최종 평균 성능] ===")
    print(pd.DataFrame(all_metrics).mean())


if __name__ == '__main__':
    main()