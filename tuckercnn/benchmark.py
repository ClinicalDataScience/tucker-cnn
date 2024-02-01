import os

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from nnunetv2.inference.predict_from_raw_data import load_what_we_need

from tuckercnn.tucker import DecompositionAgent
from tuckercnn.utils import streamline_nnunet_architecture
from tuckercnn.timer import Timer


@torch.no_grad()
def exec_benchmark(benchmark_args: dict, tucker_args: dict) -> None:
    # Load Network
    # ----------------------------------------------------------------------------------
    mostly_useless_stuff = load_what_we_need(
        model_training_output_dir=benchmark_args['nnunet_path'],
        use_folds=[0],
        checkpoint_name='checkpoint_final.pth',
    )
    network = mostly_useless_stuff[-2]

    if benchmark_args['load_params']:
        params = mostly_useless_stuff[0][0]
        network.load_state_dict(params)

    if benchmark_args['apply_tucker']:
        network = DecompositionAgent(
            tucker_args=tucker_args,
            ckpt_path=benchmark_args['ckpt_path'],
            save_model=benchmark_args['save_model'],
            load_model=benchmark_args['load_model'],
        )(network)

    network = streamline_nnunet_architecture(network)
    network = network.to(benchmark_args['device'])
    network.eval()

    # Count FLOPs
    # ----------------------------------------------------------------------------------
    x = torch.rand(
        benchmark_args['batch_size'],
        *benchmark_args['patch_size'],
        device=benchmark_args['device'],
    )

    flops = FlopCountAnalysis(network, x)
    print(flop_count_table(flops))
    print(f'Total Giga Flops: {flops.total() / 1000 ** 3:.3f}G')
    print(f'Number of parameters: {parameter_count(network)[""] / 1e6:.3f}M')

    # Measure execution time
    # ----------------------------------------------------------------------------------
    if benchmark_args['compile']:
        network = torch.compile(network)

    print('Measuring forward pass time ...')
    Timer.use_cuda = False if benchmark_args['device'] == 'cpu' else True

    with torch.autocast(benchmark_args['device'], enabled=benchmark_args['autocast']):
        for i in range(benchmark_args['eval_passes'] + 1):
            with Timer():
                network(x)

    Timer.report()
