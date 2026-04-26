import hydra
import pytest


def test_configured_datamodule_smoke(cfg_train) -> None:
    """Smoke-test the real configured datamodule once the init stage creates it."""
    if "data" not in cfg_train:
        pytest.skip("default training config does not define a data target yet")

    datamodule = hydra.utils.instantiate(cfg_train.data)

    assert datamodule is not None
    assert hasattr(datamodule, "prepare_data")
    assert hasattr(datamodule, "setup")
    assert callable(datamodule.train_dataloader)
    assert callable(datamodule.val_dataloader)
    assert callable(datamodule.test_dataloader)
