"""This file prepares config fixtures for other tests."""

import os

# Force PyTorch to not see CUDA devices. This avoids crashes from
# old NVIDIA drivers where torch.cuda.is_available() returns True
# but cuda_init() fails (known issue in PyTorch 2.x + Lightning).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

PROJECT_ROOT = rootutils.find_root(search_from=__file__, indicator=".project-root")
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
DEFAULT_CONFIG_NAME = "config.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / DEFAULT_CONFIG_NAME


def pytest_configure(config: pytest.Config) -> None:
    """Ensures CUDA is hidden before any test modules are collected/imported."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pytest_sessionstart(session: pytest.Session) -> None:
    """Patch torch.load to use weights_only=False.

    PyTorch 2.6+ changed the default of torch.load from False to True,
    which blocks loading checkpoints with custom classes (e.g. SimpleDenseNet).
    Lightning hardcodes weights_only=True internally. We patch the source
    function cloud_io._load so all child processes (ddp_spawn) also get the fix.
    """
    import torch
    from lightning.fabric.utilities import cloud_io

    # Patch torch.load directly
    _original_torch_load = torch.load

    def _patched_torch_load(f, *args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(f, *args, **kwargs)

    torch.load = _patched_torch_load

    # Patch the source function. All modules (torch_io.pl_load, etc.) import
    # this same function object, so patching it here fixes all call sites
    # including those in spawned child processes.
    _original_load = cloud_io._load

    def _patched_load(path_or_url, map_location=None, weights_only=None):
        return _original_load(path_or_url, map_location=map_location, weights_only=False)

    cloud_io._load = _patched_load


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    if not DEFAULT_CONFIG_PATH.exists():
        pytest.skip(
            f"{DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)} has not been created by the init stage yet"
        )

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name=DEFAULT_CONFIG_NAME, return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(PROJECT_ROOT)
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    if not DEFAULT_CONFIG_PATH.exists():
        pytest.skip(
            f"{DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)} has not been created by the init stage yet"
        )

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name=DEFAULT_CONFIG_NAME, return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(PROJECT_ROOT)
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
