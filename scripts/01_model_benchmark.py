import os

from nnunetv2.inference.predict_from_raw_data import load_what_we_need
import torch
from torchinfo import summary

from tuckercnn.utils import Timer
from tuckercnn.tucker import DecompositionAgent

MODEL_OUTPUT_DIR = (
    '.totalsegmentator/nnunet/results/'
    'Dataset297_TotalSegmentator_total_3mm_1559subj/'
    'nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres'
)

TUCKER_ARGS = {
    'rank_mode': 'relative',
    'rank_factor': 1 / 3,
    'rank_min': 16,
    'decompose': False,
    'verbose': False,
}


@torch.no_grad()
def main() -> None:
    mostly_useless_stuff = load_what_we_need(
        model_training_output_dir=os.path.join(os.environ['HOME'], MODEL_OUTPUT_DIR),
        use_folds=[0],
        checkpoint_name='checkpoint_final.pth',
    )

    params = mostly_useless_stuff[0][0]
    network = mostly_useless_stuff[-2]

    network.load_state_dict(params)
    network = network.cuda()
    network.eval()

    network = DecompositionAgent(tucker_args=TUCKER_ARGS)(network)
    x = torch.rand(5, 1, 112, 112, 128).cuda()

    summary(network, (1, 1, 112, 112, 128))

    with torch.autocast('cuda'):
        for i in range(21):
            with Timer():
                network(x)

    Timer.report()


if __name__ == '__main__':
    main()
