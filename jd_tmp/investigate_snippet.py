import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from nnunetv2.inference.predict_from_raw_data import load_what_we_need

from tuckercnn.tucker import DecompositionAgent
from tuckercnn.utils import streamline_nnunet_architecture
from tuckercnn.timer import Timer


@torch.no_grad()
def exec_benchmark(
        benchmark_args: dict, tucker_args: dict, verbose: bool = True
) -> dict:
    Timer.verbose = verbose

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

    def printnorm(self, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        if self.__class__.__name__ == "Conv3d":
            print('Inside ' + self.jname + ' forward ' + self.__class__.__name__, input[0].shape, output.shape)

    for n, m in network.named_modules():
        setattr(m, "jname", n)
        m.register_forward_hook(printnorm)

    network = network.to(benchmark_args['device'])
    network.eval()

    # Count FLOPs
    # ----------------------------------------------------------------------------------
    x = torch.rand(
        benchmark_args['batch_size'],
        *benchmark_args['patch_size'],
        device=benchmark_args['device'],
    )

    y = network(x)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./logs')
    writer.add_graph(network, x)
    writer.close()
    return 0

    flops = FlopCountAnalysis(network, x)
    flops.unsupported_ops_warnings(enabled=verbose)

    flops_total = flops.total() / 1000 ** 3
    num_params = parameter_count(network)[""] / 1e6

    if verbose:
        print(flop_count_table(flops))
        print(f'Total Giga Flops: {flops_total:.3f}G')
        print(f'Number of parameters: {num_params:.3f}M')
        print('Measuring forward pass time ...')

    # Measure execution time
    # ----------------------------------------------------------------------------------
    if benchmark_args['compile']:
        network = torch.compile(network)

    Timer.use_cuda = False if benchmark_args['device'] == 'cpu' else True

    with torch.autocast(benchmark_args['device'], enabled=benchmark_args['autocast']):
        for i in range(benchmark_args['eval_passes'] + 1):
            with Timer():
                network(x)

    if verbose:
        Timer.report()

    exec_time_mean = float(np.mean(Timer.get_exec_times()))
    exec_time_std = float(np.std(Timer.get_exec_times()))
    device_name = (
        torch.cuda.get_device_name() if benchmark_args['device'] == 'cuda' else ''
    )

    Timer.reset()
    return {
        'g_flops': flops_total,
        'm_params': num_params,
        'exec_time_mean': exec_time_mean,
        'exec_time_std': exec_time_std,
        'device_name': device_name,
    }
