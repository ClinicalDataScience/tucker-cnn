import time
import multiprocessing
import shutil
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union

import numpy as np
import tensorly
import torch
from torch import nn
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    isfile,
    maybe_mkdir_p,
    save_json,
)
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction_from_softmax
from nnunetv2.inference.predict_from_raw_data import PreprocessAdapter, \
    auto_detect_available_folds, load_what_we_need
from nnunetv2.inference.sliding_window_prediction import (
    predict_sliding_window_return_logits,
    compute_gaussian,
)
from nnunetv2.utilities.file_path_utilities import (
    get_output_folder,
    should_i_save_to_file,
    check_workers_busy,
)
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


from utils import eprint
from tucker import Decomposer

tensorly.set_backend('numpy')


class DummyFile(object):
   def write(self, x): pass
   def flush(self): pass


def predict_from_raw_data(
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder: str,
        model_training_output_dir: str,
        use_folds: Union[Tuple[int, ...], str] = None,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_gpu: bool = True,
        verbose: bool = True,
        save_probabilities: bool = False,
        overwrite: bool = True,
        checkpoint_name: str = 'checkpoint_final.pth',
        num_processes_preprocessing: int = default_num_processes,
        num_processes_segmentation_export: int = default_num_processes,
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0,
        device: torch.device = torch.device('cuda'),
):
    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n"
    )

    if device.type == 'cuda':
        device = torch.device(
            type='cuda', index=0
        )  # set the desired GPU with CUDA_VISIBLE_DEVICES!

    if device.type != 'cuda':
        perform_everything_on_gpu = False

    # let's store the input arguments so that its clear what was used to generate the prediction
    my_init_kwargs = {}
    # for k in inspect.signature(predict_from_raw_data).parameters.keys():
    #    my_init_kwargs[k] = locals()[k]
    my_init_kwargs = deepcopy(
        my_init_kwargs
    )  # let's not unintentionally change anything in-place. Take this as a
    # safety precaution.
    recursive_fix_for_json_export(my_init_kwargs)
    maybe_mkdir_p(output_folder)
    save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

    if use_folds is None:
        use_folds = auto_detect_available_folds(
            model_training_output_dir, checkpoint_name
        )

    # load all the stuff we need from the model_training_output_dir
    (
        parameters,
        configuration_manager,
        inference_allowed_mirroring_axes,
        plans_manager,
        dataset_json,
        network,
        trainer_name,
    ) = load_what_we_need(model_training_output_dir, use_folds, checkpoint_name)

    # check if we need a prediction from the previous stage
    if configuration_manager.previous_stage_name is not None:
        if folder_with_segs_from_prev_stage is None:
            print(
                f'WARNING: The requested configuration is a cascaded model and requires predctions from the '
                f'previous stage! folder_with_segs_from_prev_stage was not provided. Trying to run the '
                f'inference of the previous stage...'
            )
            folder_with_segs_from_prev_stage = join(
                output_folder, f'prediction_{configuration_manager.previous_stage_name}'
            )
            predict_from_raw_data(
                list_of_lists_or_source_folder,
                folder_with_segs_from_prev_stage,
                get_output_folder(
                    plans_manager.dataset_name,
                    trainer_name,
                    plans_manager.plans_name,
                    configuration_manager.previous_stage_name,
                ),
                use_folds,
                tile_step_size,
                use_gaussian,
                use_mirroring,
                perform_everything_on_gpu,
                verbose,
                False,
                overwrite,
                checkpoint_name,
                num_processes_preprocessing,
                num_processes_segmentation_export,
                None,
                num_parts=num_parts,
                part_id=part_id,
                device=device,
            )

    # sort out input and output filenames
    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(
            list_of_lists_or_source_folder, dataset_json['file_ending']
        )
    print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
    caseids = [
        os.path.basename(i[0])[: -(len(dataset_json['file_ending']) + 5)]
        for i in list_of_lists_or_source_folder
    ]
    print(
        f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)'
    )
    print(f'There are {len(caseids)} cases that I would like to predict')

    output_filename_truncated = [join(output_folder, i) for i in caseids]
    seg_from_prev_stage_files = [
        join(folder_with_segs_from_prev_stage, i + dataset_json['file_ending'])
        if folder_with_segs_from_prev_stage is not None
        else None
        for i in caseids
    ]
    # remove already predicted files form the lists
    if not overwrite:
        tmp = [
            isfile(i + dataset_json['file_ending']) for i in output_filename_truncated
        ]
        not_existing_indices = [i for i, j in enumerate(tmp) if not j]

        output_filename_truncated = [
            output_filename_truncated[i] for i in not_existing_indices
        ]
        list_of_lists_or_source_folder = [
            list_of_lists_or_source_folder[i] for i in not_existing_indices
        ]
        seg_from_prev_stage_files = [
            seg_from_prev_stage_files[i] for i in not_existing_indices
        ]
        print(
            f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
            f'That\'s {len(not_existing_indices)} cases.'
        )
        # caseids = [caseids[i] for i in not_existing_indices]

    # placing this into a separate function doesnt make sense because it needs so many input variables...
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    # hijack batchgenerators, yo
    # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    # way we don't have to reinvent the wheel here.
    num_processes = max(
        1, min(num_processes_preprocessing, len(list_of_lists_or_source_folder))
    )
    ppa = PreprocessAdapter(
        list_of_lists_or_source_folder,
        seg_from_prev_stage_files,
        preprocessor,
        output_filename_truncated,
        plans_manager,
        dataset_json,
        configuration_manager,
        num_processes,
    )
    mta = MultiThreadedAugmenter(
        ppa, NumpyToTensor(), num_processes, 1, None, pin_memory=device.type == 'cuda'
    )
    # mta = SingleThreadedAugmenter(ppa, NumpyToTensor())

    # precompute gaussian
    inference_gaussian = torch.from_numpy(
        compute_gaussian(configuration_manager.patch_size)
    ).half()
    if perform_everything_on_gpu:
        inference_gaussian = inference_gaussian.to(device)

    # num seg heads is needed because we need to preallocate the results in predict_sliding_window_return_logits
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads

    # go go go
    # spawn allows the use of GPU in the background process in case somebody wants to do this. Not recommended. Trust me.
    # export_pool = multiprocessing.get_context('spawn').Pool(num_processes_segmentation_export)
    # export_pool = multiprocessing.Pool(num_processes_segmentation_export)
    with multiprocessing.get_context("spawn").Pool(
            num_processes_segmentation_export
    ) as export_pool:
        network = network.to(device)

        r = []
        with torch.no_grad():
            for preprocessed in mta:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                print(f'\nPredicting {os.path.basename(ofile)}:')
                print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')

                properties = preprocessed['data_properites']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_busy(
                    export_pool, r, allowed_num_queued=len(export_pool._pool)
                )
                while not proceed:
                    sleep(1)
                    proceed = not check_workers_busy(
                        export_pool, r, allowed_num_queued=len(export_pool._pool)
                    )

                # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
                # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
                # things a lot faster for some datasets.
                prediction = None
                overwrite_perform_everything_on_gpu = perform_everything_on_gpu
                if perform_everything_on_gpu:
                    try:
                        for params in parameters:
                            network.load_state_dict(params)
                            network = Decomposer()(network)

                            if prediction is None:
                                prediction = predict_sliding_window_return_logits(
                                    network,
                                    data,
                                    num_seg_heads,
                                    configuration_manager.patch_size,
                                    mirror_axes=inference_allowed_mirroring_axes
                                    if use_mirroring
                                    else None,
                                    tile_step_size=tile_step_size,
                                    use_gaussian=use_gaussian,
                                    precomputed_gaussian=inference_gaussian,
                                    perform_everything_on_gpu=perform_everything_on_gpu,
                                    verbose=verbose,
                                    device=device,
                                )
                            else:
                                prediction += predict_sliding_window_return_logits(
                                    network,
                                    data,
                                    num_seg_heads,
                                    configuration_manager.patch_size,
                                    mirror_axes=inference_allowed_mirroring_axes
                                    if use_mirroring
                                    else None,
                                    tile_step_size=tile_step_size,
                                    use_gaussian=use_gaussian,
                                    precomputed_gaussian=inference_gaussian,
                                    perform_everything_on_gpu=perform_everything_on_gpu,
                                    verbose=verbose,
                                    device=device,
                                )

                        if len(parameters) > 1:
                            prediction /= len(parameters)

                    except RuntimeError:
                        print(
                            'Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                            'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...'
                        )
                        print('Error:')
                        traceback.print_exc()
                        prediction = None
                        overwrite_perform_everything_on_gpu = False

                if prediction is None:
                    for params in parameters:
                        network.load_state_dict(params)
                        if prediction is None:
                            prediction = predict_sliding_window_return_logits(
                                network,
                                data,
                                num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes
                                if use_mirroring
                                else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                                verbose=verbose,
                                device=device,
                            )
                        else:
                            prediction += predict_sliding_window_return_logits(
                                network,
                                data,
                                num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes
                                if use_mirroring
                                else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                                verbose=verbose,
                                device=device,
                            )
                    if len(parameters) > 1:
                        prediction /= len(parameters)

                print('Prediction done, transferring to CPU if needed')
                prediction = prediction.to('cpu').numpy()

                if should_i_save_to_file(prediction, r, export_pool):
                    print(
                        'output is either too large for python process-process communication or all export workers are '
                        'busy. Saving temporarily to file...'
                    )
                    np.save(ofile + '.npy', prediction)
                    prediction = ofile + '.npy'

                # this needs to go into background processes
                # export_prediction(prediction, properties, configuration_name, plans, dataset_json, ofile,
                #                   save_probabilities)
                print(
                    'sending off prediction to background worker for resampling and export'
                )
                r.append(
                    export_pool.starmap_async(
                        export_prediction_from_softmax,
                        (
                            (
                                prediction,
                                properties,
                                configuration_manager,
                                plans_manager,
                                dataset_json,
                                ofile,
                                save_probabilities,
                            ),
                        ),
                    )
                )
                print(f'done with {os.path.basename(ofile)}')
        [i.get() for i in r]

    # we need these two if we want to do things with the predictions like for example apply postprocessing
    shutil.copy(
        join(model_training_output_dir, 'dataset.json'),
        join(output_folder, 'dataset.json'),
    )
    shutil.copy(
        join(model_training_output_dir, 'plans.json'), join(output_folder, 'plans.json')
    )


def maybe_mirror_and_predict(network: nn.Module, x: torch.Tensor, mirror_axes: Tuple[int, ...] = None) \
        -> torch.Tensor:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    prediction = network(x)
    end.record()

    torch.cuda.synchronize()
    eprint(start.elapsed_time(end))

    if mirror_axes is not None:
        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

        num_predictons = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2,))), (2,))
        if 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3,))), (3,))
        if 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (4,))), (4,))
        if 0 in mirror_axes and 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3))), (2, 3))
        if 0 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 4))), (2, 4))
        if 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3, 4))), (3, 4))
        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
        prediction /= num_predictons
    return prediction