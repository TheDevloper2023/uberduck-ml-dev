from uberduck_ml_dev.trainer.base import TTSTrainer
import torch
import math
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
from uberduck_ml_dev.trainer.base import DEFAULTS as TRAINER_DEFAULTS


class TestTrainer:
    def test_trainer_base(self):

        config = TRAINER_DEFAULTS.values()

        params = dict(
            checkpoint_name="test",
            checkpoint_path="test_checkpoint",
            cudnn_enabled=True,
            log_dir="this/is/a/test",
        )
        config.update(params)
        hparams = HParams(**config)
        trainer = TTSTrainer(hparams)
        assert trainer.hparams == hparams

        assert trainer.cudnn_enabled == True
        mel = torch.load("tests/fixtures/stevejobs-1.pt")
        audio = trainer.sample(mel)
        assert audio.size(0) == 1


class TestTacotron2Trainer:

    # NOTE (Sam): this test could be made twice as fast by only running a single epoch,.
    # since as it is, the second gradient step is only useful for evaluating the loss
    def test_gradient_step(self, lj_trainer):

        torch.manual_seed(1234)
        lj_trainer.train()

        # NOTE (Sam): this number was taken from master on 8/24/22.
        train_loss_start = 0.320
        train_loss_4_datapoints_1_iteration = 0.319

        assert math.isclose(lj_trainer.loss[0], train_loss_start, abs_tol=5e-4)

        assert math.isclose(
            lj_trainer.loss[1], train_loss_4_datapoints_1_iteration, abs_tol=5e-4
        )
