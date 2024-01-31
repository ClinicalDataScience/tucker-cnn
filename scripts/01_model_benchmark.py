import os

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from nnunetv2.inference.predict_from_raw_data import load_what_we_need

from tuckercnn.tucker import DecompositionAgent
from tuckercnn.utils import streamline_nnunet_architecture
from tuckercnn.timer import Timer

MODEL_OUTPUT_DIR = (
    '.totalsegmentator/nnunet/results/'
    'Dataset297_TotalSegmentator_total_3mm_1559subj/'
    'nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres'
)

TUCKER_ARGS = {
    'rank_mode': 'relative',
    'rank_factor': 1 / 3,
    'rank_min': None,
    'decompose': False,
    'verbose': False,
}

BENCHMARK_ARGS = {
    'patch_size': (1, 112, 112, 128),
    'batch_size': 1,
    'device': 'cudaq',
    'load_params': False,
    'apply_tucker': True,
    'autocast': False,
    'compile': False,
    'eval_passes': 10,
}


@torch.no_grad()
def main() -> None:
    # Load Network
    # ----------------------------------------------------------------------------------
    mostly_useless_stuff = load_what_we_need(
        model_training_output_dir=os.path.join(os.environ['HOME'], MODEL_OUTPUT_DIR),
        use_folds=[0],
        checkpoint_name='checkpoint_final.pth',
    )
    network = mostly_useless_stuff[-2]

    if BENCHMARK_ARGS['load_params']:
        params = mostly_useless_stuff[0][0]
        network.load_state_dict(params)

    if BENCHMARK_ARGS['apply_tucker']:
        network = DecompositionAgent(tucker_args=TUCKER_ARGS)(network)

    network = streamline_nnunet_architecture(network)
    network = network.to(BENCHMARK_ARGS['device'])
    network.eval()

    # Count FLOPs
    # ----------------------------------------------------------------------------------
    x = torch.rand(
        BENCHMARK_ARGS['batch_size'],
        *BENCHMARK_ARGS['patch_size'],
        device=BENCHMARK_ARGS['device'],
    )

    flops = FlopCountAnalysis(network, x)
    print(flop_count_table(flops))
    print(f'Total Giga Flops: {flops.total() / 1000 ** 3:.3f}G')
    print(f'Number of parameters: {parameter_count(network)[""] / 1e6:.3f}M')

    # Measure execution time
    # ----------------------------------------------------------------------------------
    if BENCHMARK_ARGS['compile']:
        network = torch.compile(network)

    print('Measuring forward pass time ...')
    Timer.use_cuda = False if BENCHMARK_ARGS['device'] == 'cpu' else True

    with torch.autocast(BENCHMARK_ARGS['device'], enabled=BENCHMARK_ARGS['autocast']):
        for i in range(BENCHMARK_ARGS['eval_passes'] + 1):
            with Timer():
                network(x)

    Timer.report()


if __name__ == '__main__':
    main()
