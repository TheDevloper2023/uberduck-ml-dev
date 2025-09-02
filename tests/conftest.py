import torch
import tempfile
import pytest
import os

import gdown
from scipy.io import wavfile
import numpy as np
import torch

from uberduck_ml_dev.models.tacotron2 import DEFAULTS as TACOTRON2_DEFAULTS
from uberduck_ml_dev.models.tacotron2 import Tacotron2
from uberduck_ml_dev.trainer.tacotron2 import (
    Tacotron2Trainer,
    DEFAULTS as TACOTRON2_TRAINER_DEFAULTS,
)
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams

# NOTE (Sam): move to Tacotron2 model and remove from Uberduck repo.
def _load_tacotron_uninitialized(overrides=None):
    overrides = overrides or {}
    defaults = dict(**TACOTRON2_DEFAULTS.values())
    defaults.update(overrides)
    hparams = HParams(**defaults)
    return Tacotron2(hparams)


@pytest.fixture(scope="session")
def lj_speech_tacotron2_file():
    # NOTE (Sam): A canonical LJ statedict used in our warm starting notebook.
    url = "https://drive.google.com/uc?id=1qgEwtL53oFsdllM14FRZncgnARnAGInO"
    output_file = tempfile.NamedTemporaryFile()
    gdown.download(url, output_file.name, quiet=False)
    return output_file


@pytest.fixture
def lj_speech_tacotron2(lj_speech_tacotron2_file):

    # NOTE (Sam): this override should no longer be necessary.
    device = "cpu"
    config_overrides = {}
    config_overrides["cudnn_enabled"] = device != "cpu"
    _model = _load_tacotron_uninitialized(config_overrides)
    checkpoint = torch.load(lj_speech_tacotron2_file.name, map_location=device)
    _model.from_pretrained(model_dict=checkpoint["state_dict"], device=device)

    return _model


@pytest.fixture
def sample_inference_spectrogram():
    # NOTE (Sam): made in Uberduck container using current test code in test_stft_seed.
    inference_spectrogram = torch.load(
        os.path.join(os.path.dirname(__file__), "fixtures/sample_spectrogram.pt")
    )
    return inference_spectrogram


@pytest.fixture
def sample_inference_tf_spectrogram():
    # NOTE (Sam): made with above at timestep 111 and text = "I, Sam, am a very bad boy."
    inference_spectrogram = torch.load(
        os.path.join(os.path.dirname(__file__), "fixtures/sample_spectrogram_tf.pt")
    )

    return inference_spectrogram


@pytest.fixture()
def lj_trainer(lj_speech_tacotron2_file):

    # NOTE (Sam): It may be nicer to specify trainer here and test-specific parameters (e.g. data) in test itself.
    config = TACOTRON2_TRAINER_DEFAULTS.values()
    params = dict(
        warm_start_name=lj_speech_tacotron2_file.name,
        training_audiopaths_and_text=os.path.join(
            os.path.dirname(__file__), "fixtures/ljtest/list_small.txt"
        ),
        val_audiopaths_and_text=os.path.join(
            os.path.dirname(__file__), "fixtures/ljtest/list_small.txt"
        ),
        checkpoint_name="test",
        checkpoint_path="test_checkpoint",
        epochs=2,
        log_dir="",
        debug=True,
        batch_size=4,
        learning_rate=1e-4,
        # NOTE (Sam): this effects the reduction in loss in the gradient descent,
        # so we need a separate test of validation and logging code.
        is_validate=False,
    )
    config.update(params)
    hparams = HParams(**config)

    trainer = Tacotron2Trainer(hparams, rank=0, world_size=1)

    return trainer
