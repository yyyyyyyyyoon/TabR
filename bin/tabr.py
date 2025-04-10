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

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from torch import Tensor
from tqdm import tqdm
import data
from torch.utils.tensorboard import SummaryWriter
import lib
from data import preprocess_data
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from loguru import logger
from math import sqrt

from typing import Dict, Any
KWArgs = Dict[str, Any]
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

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
        *,
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
            # NOTE: during evaluation, candidate keys can be computed just once, which
            # looks like an easy opportunity for optimization. However:
            # - if your dataset is small or/and the encoder is just a linear layer
            #   (no embeddings and encoder_n_blocks=0), then encoding candidates
            #   is not a bottleneck.
            # - implementing this optimization makes the code complex and/or unobvious,
            #   because there are many things that should be taken into account:
            #     - is the input coming from the "train" part?
            #     - is self.training True or False?
            #     - is PyTorch autograd enabled?
            #     - is saving and loading checkpoints handled correctly?
            # This is why we do not implement this optimization.

            # When memory_efficient is True, this potentially heavy computation is
            # performed without gradients.
            # Later, it is recomputed with gradients only for the context objects.
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
            # NOTE: here, we add the training batch back to the candidates after the
            # function `apply_model` removed them. The further code relies
            # on the fact that the first batch_size candidates come from the
            # training batch.
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        # >>>
        # The search below is optimized for larger datasets and is significantly faster
        # than the naive solution (keep autograd on + manually compute all pairwise
        # squared L2 distances + torch.topk).
        # For smaller datasets, however, the naive solution can actually be faster.
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == 'cuda'
                    else faiss.IndexFlatL2(d_main)
                )
            # Updating the index is much faster than creating a new one.
            self.search_index.reset()
            self.search_index.add(candidate_k)  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k, context_size + (1 if is_train else 0)
            )
            if is_train:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
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

        # In theory, when autograd is off, the distances obtained during the search
        # can be reused. However, this is not a bottleneck, so let's keep it simple
        # and use the same code to compute `similarities` during both
        # training and evaluation.
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


def main() -> None:
    # 로그 디렉토리 설정
    log_dir = "logs/run1"  # 문자열 경로로 설정

    # 디렉토리가 없으면 생성
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # SummaryWriter 초기화
    writer = SummaryWriter(log_dir=log_dir)

    # 데이터 전처리
    splits = preprocess_data(data.file_paths)
    # 모델 생성 및 학습
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        print("Error: No dataset name provided.")
        sys.exit(1)

    # 전처리된 데이터 가져오기
    if dataset_name not in splits:
        print(f"Error: Dataset '{dataset_name}' not found in splits.")
        sys.exit(1)

    X_train = splits[dataset_name]["X_train"]
    X_test = splits[dataset_name]["X_test"]
    y_train = splits[dataset_name]["y_train"]
    y_test = splits[dataset_name]["y_test"]

    # PyTorch Tensor로 변환
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_train_tensor = torch.from_numpy(numpy.array(y_train)).long()
    y_test_tensor = torch.from_numpy(numpy.array(y_test)).long()

    # model 생성
    model = Model(
        n_num_features=X_train.shape[1],
        n_bin_features=1,
        cat_cardinalities=[],
        n_classes=2,
        num_embeddings=None,  # 또는 필요한 딕셔너리 제공
        d_main=128,
        d_multiplier=2.0,
        encoder_n_blocks=3,
        predictor_n_blocks=2,
        mixer_normalization='auto',
        context_dropout=0.1,
        dropout0=0.2,
        dropout1='dropout0',  # 또는 float 값 제공
        normalization='BatchNorm1d',
        activation='ReLU',
        memory_efficient=False,
        candidate_encoding_batch_size=None,
    )
    model.to(device)

    X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)

    C = Config(
        seed=42,
        data={
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        },
        model={"model_name": "TabR"},
        context_size=10,
        optimizer={"optimizer_name": "Adam"},
        batch_size=32,
        patience=5,
        n_epochs=10
    )

    #모델 훈련
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # 학습 루프
    for epoch in range(10):  # 학습 에포크 수
        model.train()
        optimizer.zero_grad()

        # 입력 데이터를 모델에 맞게 구성
        x_ = {'num': X_train_tensor}
        y = y_train_tensor

        # y의 형상 확인
        batch_size = y.shape[0]

        candidate_x_ = {'num': X_train_tensor}  # 후보 데이터 (예시로 학습 데이터 사용)
        candidate_y = y_train_tensor
        context_size = 10  # 컨텍스트 크기
        is_train = True

        output = model(x_=x_, y=y, candidate_x_=candidate_x_, candidate_y=candidate_y, context_size=context_size, is_train=is_train)

        loss = loss_fn(output, y)
        loss_value = loss.item()
        loss.backward()
        optimizer.step()

       # print(f"Epoch {epoch + 1}, Loss: {loss_value}")

    # 모델 예측
    model.eval()
    with torch.no_grad():
        # 입력 데이터 준비
        x_ = {'num': X_test_tensor}  # 입력 데이터
        candidate_x_ = {'num': X_test_tensor}  # 후보 데이터

        # 모델 호출
        predictions = model(
            x_=x_,
            y=None,
            candidate_x_=candidate_x_,
            candidate_y=y_test_tensor,  # 테스트 레이블
            context_size=10,  # 컨텍스트 크기
            is_train=False,  # 평가 모드
        )
        probs = torch.softmax(predictions, dim=1)
        positive_probs = probs[:, 1]

    # 예측값을 이진 분류 결과로 변환
    _, predicted = torch.max(predictions, dim=1)

    # 성능 평가
    def calculate_metrics(y_true, y_pred):
        f1 = f1_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        TP, FP, FN, TN = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]

        PD = TP / (TP + FN) if (TP + FN) > 0 else 0
        PF = FP / (FP + TN) if (FP + TN) > 0 else 0
        FIR = (PD - f1) / PD if PD > 0 else 0
        Blance = 1 - (sqrt((0 - PF) ** 2 + (1 - PD) ** 2) / sqrt(2))

        return {
            "PD": PD,
            "PF": PF,
            "Blance" : Blance,
            "FIR": FIR
        }

    metrics = calculate_metrics(
        y_test_tensor.cpu().numpy(),
        predicted.cpu().numpy()
    )

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


    epoch = 0
    eval_batch_size = min(32, len(X_test_tensor))
    chunk_size = None
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)  # type: ignore[code]

    def get_Xy(part: str, idx: Optional[Tensor]) -> tuple[dict[str, Tensor], Tensor]:
        if part == 'train':
            X = X_train_tensor
            y = y_train_tensor
        elif part == 'test':
            X = X_test_tensor
            y = y_test_tensor
        else:
            raise ValueError("Invalid part")

        if idx is not None:
            X = X[idx]
            y = y[idx]

        return {'num': X}, y

    train_size = len(y_train_tensor)
    train_indices = torch.arange(train_size, device=device)


    @torch.inference_mode()
    def evaluate(parts: list[str], eval_batch_size: int):
        model.eval()
        predictions = {}
        parts = ['train', 'test']

        for part in parts:
            if part == 'train':
                X = X_train_tensor
                y = y_train_tensor
            elif part == 'test':
                X = X_test_tensor
                y = y_test_tensor

            predictions[part] = []
            for i in range(0, len(X), eval_batch_size):
                batch_X = X[i:i + eval_batch_size]
                batch_y = y[i:i + eval_batch_size]

                x_ = {'num': batch_X}
                candidate_x_ = {'num': batch_X}  # 후보 데이터 (예시로 학습 데이터 사용)
                candidate_y = batch_y
                context_size = 10  # 컨텍스트 크기
                is_train = False

                output = model(
                    x_=x_,
                    y=None,
                    candidate_x_=candidate_x_,
                    candidate_y=candidate_y,
                    context_size=context_size,
                    is_train=is_train,
                )
                predictions[part].append(output.cpu().numpy())

            predictions[part] = numpy.concatenate(predictions[part])

        metrics = {}
        for part in parts:
            if part == 'train':
                y = y_train_tensor
            elif part == 'test':
                y = y_test_tensor

            _, predicted = torch.max(torch.from_numpy(predictions[part]), dim=1)
            accuracy = (predicted == y).sum().item() / len(y)
            metrics[part] = {'accuracy': accuracy}

        return metrics, predictions, eval_batch_size

    def save_checkpoint():
        lib.dump_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'random_state': delu.random.get_state(),
                'progress': progress,
                'timer': timer,
                'training_log': training_log,
            },
            output,
        )
        lib.backup_output(output)

    print()
    timer = lib.run_timer()
    while epoch < C.n_epochs:
        print(f'[...] {lib.try_get_relative_path(output)} | {timer}')

        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            lib.make_random_batches(train_size, C.batch_size, device),
            desc=f'Epoch {epoch}',
        ):
            # batch_idx에 해당하는 데이터를 모델에 맞게 준비
            batch_x = X_train_tensor[batch_idx]
            batch_y = y_train_tensor[batch_idx]
            y = batch_y

            x_ = {'num': batch_x}
            candidate_x_ = {'num': X_train_tensor}  # 후보 데이터 (예시로 학습 데이터 사용)
            candidate_y = y_train_tensor
            context_size = 10  # 컨텍스트 크기
            is_train = True
            # 모델에 입력을 제공하고 손실 계산
            output = model(x_=x_, y=y, candidate_x_=candidate_x_, candidate_y=candidate_y, context_size=context_size, is_train=False)
            loss = loss_fn(output, batch_y)

            loss, new_chunk_size = lib.train_step(
                optimizer,
                lambda idx: loss_fn(model(
                    {'num': X_train_tensor[idx]},
                    y_train_tensor[idx],
                    {'num': X_train_tensor},  # 후보 데이터 (예시로 학습 데이터 사용)
                    y_train_tensor,
                    context_size,
                    True
                ), y_train_tensor[idx]),
                batch_idx,
                chunk_size or C.batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f'chunk_size = {chunk_size}')

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )
        lib.print_metrics(mean_loss, metrics)
        training_log.append(
            {'epoch-losses': epoch_losses, 'metrics': metrics, 'time': timer()}
        )
        writer.add_scalars('loss', {'train': mean_loss}, epoch, timer())
        for part in metrics:
            writer.add_scalars('score', {part: metrics[part]['score']}, epoch, timer())

        progress.update(metrics['val']['score'])
        if progress.success:
            lib.celebrate()
            save_checkpoint()
            lib.dump_predictions(predictions, output)

        elif progress.fail or not lib.are_valid_predictions(predictions):
            break

        epoch += 1
        print()

    # >>> finish
    model.load_state_dict(lib.load_checkpoint(output)['model'])
    lib.dump_predictions(predictions, output)
    save_checkpoint()


#if __name__ == '__main__':
 #   lib.configure_libraries()
  #  lib.run_Function_cli(main)

if __name__ == '__main__':
    main()
