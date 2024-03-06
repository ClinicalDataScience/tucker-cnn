from typing import Optional

from dataclasses import dataclass

DEFAULT_CONFIG_DICT = {
    'tucker_args': {
        'rank_mode': 'relative',
        'rank_factor': 1 / 3,
        'rank_min': 16,
        'decompose': True,
        'verbose': True,
    },
    'apply_tucker': True,
    'inference_bs': 1,
    'ckpt_path': '',
    'save_model': False,
    'load_model': False,
}


@dataclass
class MonkeyConfig:
    tucker_args: Optional[dict]
    apply_tucker: bool
    inference_bs: int
    ckpt_path: str
    save_model: bool
    load_model: bool


@dataclass
class MonkeyTrainConfig:
    dataset_id: str
    epochs: int
    learning_rate: float
    optimizer: str
    tracing_patch_size: tuple[int, ...] = (1, 1, 128, 128, 128)
