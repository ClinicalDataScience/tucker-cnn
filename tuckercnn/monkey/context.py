from typing import Optional

from nnunetv2.inference import predict_from_raw_data
from nnunetv2.inference import sliding_window_prediction
from nnunetv2.run import run_training
from totalsegmentator import libs

from tuckercnn.monkey import monkey_inference, monkey_train
from tuckercnn.monkey.config import MonkeyConfig, DEFAULT_CONFIG_DICT
from tuckercnn.timer import Timer

MONKEY_PATCHES = {
    (libs, 'DummyFile'): monkey_inference.DummyFile,
    (
        predict_from_raw_data,
        'predict_from_raw_data',
    ): monkey_inference.predict_from_raw_data,
    (
        sliding_window_prediction,
        'maybe_mirror_and_predict',
    ): monkey_inference.maybe_mirror_and_predict,
    (run_training, 'maybe_load_checkpoint'): monkey_train.maybe_load_checkpoint,
}


class TuckerContext:
    def __init__(self, config: Optional[dict] = None, patches: Optional[dict] = None):
        self.config = DEFAULT_CONFIG_DICT if config is None else config

        self.patches = MONKEY_PATCHES if patches is None else patches
        self.originals = {}

    def set_config(self) -> None:
        MonkeyConfig.tucker_args = self.config['tucker_args']
        MonkeyConfig.apply_tucker = self.config['apply_tucker']
        MonkeyConfig.inference_bs = self.config['inference_bs']
        MonkeyConfig.ckpt_path = self.config['ckpt_path']
        MonkeyConfig.save_model = self.config['save_model']
        MonkeyConfig.load_model = self.config['load_model']

    @staticmethod
    def restore_config() -> None:
        MonkeyConfig.tucker_args = DEFAULT_CONFIG_DICT['tucker_args']
        MonkeyConfig.apply_tucker = DEFAULT_CONFIG_DICT['apply_tucker']
        MonkeyConfig.inference_bs = DEFAULT_CONFIG_DICT['inference_bs']
        MonkeyConfig.ckpt_path = DEFAULT_CONFIG_DICT['ckpt_path']
        MonkeyConfig.save_model = DEFAULT_CONFIG_DICT['save_model']
        MonkeyConfig.load_model = DEFAULT_CONFIG_DICT['load_model']

    def __enter__(self):
        self.set_config()
        for (module, func_name), new_func in self.patches.items():
            self.originals[(module, func_name)] = getattr(module, func_name)
            setattr(module, func_name, new_func)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Timer.report()

        self.restore_config()
        for (module, func_name), original_func in self.originals.items():
            setattr(module, func_name, original_func)
