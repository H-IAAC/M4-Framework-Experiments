from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
from librep.metrics.report import ClassificationReport

sys.path.append("..")
sys.path.append(".")


# --- Actual imports ---
import pytest

import numpy as np
import pandas as pd

from experiment_executor.execute import (
    load_datasets,
    do_transform,
    do_reduce,
    do_classification,
)
from experiment_executor.utils import load_yaml
import experiment_executor.config as config
from librep.base import Estimator, Transform


@pytest.fixture
def treinable_data():
    locations = load_yaml("tests/dataset_locations.yaml")

    train_dset = load_datasets(
        dataset_locations=locations, datasets_to_load=["example.balanced[train]"]
    )
    validation_dset = load_datasets(
        dataset_locations=locations, datasets_to_load=["example.balanced[validation]"]
    )
    test_dset = load_datasets(
        dataset_locations=locations, datasets_to_load=["example.balanced[test]"]
    )

    return train_dset, validation_dset, test_dset


@pytest.fixture
def datasets(treinable_data):
    train, validation, test = treinable_data
    _datasets = dict()
    _datasets["train_dataset"] = train
    _datasets["validation_dataset"] = validation
    _datasets["test_dataset"] = test
    _datasets["reducer_dataset"] = train
    _datasets["reducer_validation_dataset"] = validation
    return _datasets


def test_load_datasets():
    locations = load_yaml("tests/dataset_locations.yaml")

    # --- Test loading a single dataset ---
    dset = load_datasets(
        dataset_locations=locations, datasets_to_load=["example.balanced[train]"]
    )
    assert len(dset) == 50
    assert dset[:][0].shape == (50, 30)

    # --- Test loading multiple datasets ---
    dset = load_datasets(
        dataset_locations=locations,
        datasets_to_load=[
            "example.balanced[train]",
            "example.balanced[validation]",
            "example.balanced[test]",
        ],
    )
    assert len(dset) == 70
    assert dset[:][0].shape == (70, 30)

    # --- Test loading multiple datasets with different features ---
    dset = load_datasets(
        dataset_locations=locations,
        datasets_to_load=[
            "example.balanced[train]",
            "example.balanced[validation]",
            "example.balanced[test]",
        ],
        features=["accel-x", "accel-y", "accel-z"],
    )
    assert len(dset) == 70
    assert dset[:][0].shape == (70, 15)

    with pytest.raises(KeyError):
        dset = load_datasets(
            dataset_locations=locations,
            datasets_to_load=[
                "xxx.balanced[train]",
            ],
        )
        
    with pytest.raises(KeyError):
        dset = load_datasets(
            dataset_locations=locations,
            datasets_to_load=[
                "example.ooo[train]",
            ],
        )
        
    with pytest.raises(KeyError):
        dset = load_datasets(
            dataset_locations=locations,
            datasets_to_load=[
                "example.balanced[what?]",
            ],
        )


def test_loader(treinable_data):
    train, validation, test = treinable_data
    assert len(train) == 50
    assert len(validation) == 10
    assert len(test) == 10


def test_dataset(datasets):
    assert len(datasets["train_dataset"]) == 50
    assert len(datasets["validation_dataset"]) == 10
    assert len(datasets["test_dataset"]) == 10
    assert len(datasets["reducer_dataset"]) == 50
    assert len(datasets["reducer_validation_dataset"]) == 10

    assert datasets["train_dataset"].num_windows == 6


def test_do_transform_identity(datasets):
    sample_10, label_10 = datasets["train_dataset"][10]
    identity_transform_config = config.TransformConfig(
        name="transform", transform="identity", kwargs=None, windowed=None
    )
    dset = do_transform(datasets, [identity_transform_config])
    assert len(dset["train_dataset"]) == 50
    assert len(dset["validation_dataset"]) == 10
    assert len(dset["test_dataset"]) == 10
    assert len(dset["reducer_dataset"]) == 50
    assert len(dset["reducer_validation_dataset"]) == 10

    np.testing.assert_array_equal(dset["train_dataset"][10][0], sample_10)
    np.testing.assert_array_equal(dset["train_dataset"][10][1], label_10)


def test_do_transform_chain_1(datasets):
    class SumTransform(Transform):
        def __init__(self, value):
            self.value = value

        def transform(self, X):
            return X + self.value

    sum1 = SumTransform(value=10)
    sum2 = SumTransform(value=5)

    sample = datasets["train_dataset"][:][0].copy()

    sum_transform_config_1 = config.TransformConfig(
        name="transform1",
        transform="WrapperTransform",
        kwargs={"obj": sum1},
        windowed=None,
    )
    sum_transform_config_2 = config.TransformConfig(
        name="transform2",
        transform="WrapperTransform",
        kwargs={"obj": sum2},
        windowed=None,
    )
    dset = do_transform(datasets, [sum_transform_config_1, sum_transform_config_2])

    np.testing.assert_array_equal(dset["train_dataset"][:][0], sample + 10 + 5)


class SimpleReducer(Transform):
    def __init__(
        self,
        expect_y: bool = False,
        expect_validation: bool = False,
    ):
        self.expect_y = expect_y
        self.expect_validation = expect_validation

    def fit(
        self,
        X,
        y=None,
        X_val=None,
        y_val=None,
        **fit_params,
    ):
        assert X is not None
        if self.expect_y:
            assert len(y) == len(X)
        if self.expect_validation:
            assert X_val is not None
        if self.expect_validation and self.expect_y:
            assert len(y_val) == len(X_val)
        return self

    def transform(self, X):
        res = np.expand_dims(np.mean(X, axis=1), axis=1)
        return res


def test_do_reduce_no_dataset_fail(datasets):
    simple_reducer = SimpleReducer()
    reducer_config = config.ReducerConfig(
        name="reducer",
        algorithm="WrapperTransform",
        kwargs={"obj": simple_reducer},
    )

    with pytest.raises(ValueError):
        do_reduce(datasets, simple_reducer, reducer_dataset_name="Invalid dataset")

    with pytest.raises(ValueError):
        do_reduce(datasets, simple_reducer, apply_only_in=["invalid row"])


@pytest.mark.parametrize(
    "expect_y, expect_validation, reduce_on, excpected_dims, apply_only_in",
    [
        # All
        (False, False, "all", 1, None),
        (True, False, "all", 1, None),
        (True, True, "all", 1, None),
        (True, True, "all", 1, ("train_dataset", "validation_dataset")), # Do not apply to test_dataset 
        # Sensor
        (False, False, "sensor", 2, None),
        (True, False, "sensor", 2, None),
        (True, True, "sensor", 2, None),
        # Axis
        (False, False, "axis", 6, None),
        (True, False, "axis", 6, None),
        (True, True, "axis", 6, None),
    ],
)
def test_do_reduce(datasets, expect_y, expect_validation, reduce_on, excpected_dims, apply_only_in):
    simple_reducer = SimpleReducer(
        expect_y=expect_y,
        expect_validation=expect_validation,
    )
    reducer_config = config.ReducerConfig(
        name="reducer",
        algorithm="WrapperTransform",
        kwargs={"obj": simple_reducer},
    )

    if expect_validation is False:
        del datasets["reducer_validation_dataset"]
        del datasets["validation_dataset"]

    original_datasets = {k: v[:][0].copy() for k, v in datasets.items()}

    new_datasets = do_reduce(
        datasets=datasets,
        reducer_config=reducer_config,
        apply_only_in=apply_only_in,
        reduce_on=reduce_on,
        use_y=expect_y
    )

    assert len(new_datasets["train_dataset"]) == 50
    assert len(new_datasets["test_dataset"]) == 10
    assert len(new_datasets["reducer_dataset"]) == 50
    if expect_validation:
        assert len(new_datasets["validation_dataset"]) == 10
        assert len(new_datasets["reducer_validation_dataset"]) == 10

    assert new_datasets["train_dataset"][:][0].shape == (50, excpected_dims)
    if apply_only_in is None:
        assert new_datasets["test_dataset"][:][0].shape == (10, excpected_dims)
    if expect_validation:
        assert new_datasets["validation_dataset"][:][0].shape == (10, excpected_dims)

    # --- untouched datasets ---
    np.testing.assert_equal(
        original_datasets["reducer_dataset"], new_datasets["reducer_dataset"][:][0]
    )

    if expect_validation:
        np.testing.assert_equal(
            original_datasets["reducer_validation_dataset"],
            new_datasets["reducer_validation_dataset"][:][0],
        )
        
    if apply_only_in is not None:
        np.testing.assert_equal(
            original_datasets["test_dataset"],
            new_datasets["test_dataset"][:][0],
        )

def test_do_reduce_fail(datasets):
    simple_reducer = SimpleReducer()
    reducer_config = config.ReducerConfig(
        name="reducer",
        algorithm="WrapperTransform",
        kwargs={"obj": simple_reducer},
    )

    with pytest.raises(ValueError):
        do_reduce(datasets, reducer_config, reducer_dataset_name="Invalid dataset")

    with pytest.raises(ValueError):
        do_reduce(datasets, reducer_config, apply_only_in=["invalid dataset"])
        
    with pytest.raises(ValueError):
        do_reduce(datasets, reducer_config, apply_only_in=["train_dataset", "invalid dataset"])
    
    with pytest.raises(ValueError):
        do_reduce(datasets, reducer_config, reduce_on="invalid")    


class SimpleEstimator(Estimator):
    def __init__(self, expect_validation: bool = False, return_class: int = 0):
        self.expect_validation = expect_validation
        self.return_class = return_class

    def fit(
        self,
        X,
        y=None,
        X_val=None,
        y_val=None,
        **fit_params,
    ):
        assert len(y) == len(X)

        if self.expect_validation:
            assert X_val is not None
            assert y_val is not None
            assert len(y_val) == len(X_val)
        return self

    def predict(self, X):
        return np.array([self.return_class] * len(X))


@pytest.mark.parametrize(
    "expect_validation",
    [
        True,
        False,
    ],
)
def test_do_classification(datasets, expect_validation):
    simple_estimator = SimpleEstimator(
        expect_validation=expect_validation,
    )
    estimator_config = config.EstimatorConfig(
        name="estimator",
        algorithm="WrapperEstimator",
        kwargs={"obj": simple_estimator},
        num_runs=1,
    )

    reporter = ClassificationReport(
        use_accuracy=True,
        use_f1_score=True,
        use_classification_report=True,
        use_confusion_matrix=True,
        plot_confusion_matrix=False,
    )

    results = do_classification(
        datasets=datasets, estimator_config=estimator_config, reporter=reporter
    )

    pass
