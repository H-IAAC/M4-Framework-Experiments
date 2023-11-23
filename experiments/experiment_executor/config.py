# Copyright © 2023 Otávio Napoli <otavio.napoli@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to 
# deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE. 

# Python imports
from dataclasses import dataclass
from typing import List, Optional

# Librep imports
from librep.base.transform import Transform
from librep.estimators import SVC, KNeighborsClassifier, RandomForestClassifier
from librep.transforms.fft import FFT
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from librep.estimators.simclr.torch.simclr_linear_estimator import  Simclr_Linear_Estimator
from librep.estimators.simclr.torch.simclr_full_estimator import Simclr_Full_Estimator
from librep.transforms.simclr import SimCLR
from librep.transforms.simclr_linear import SimCLR_linear
from librep.transforms.simclr_full import SimCLR_full

#from transforms import Spectrogram

# Third-party imports
from umap import UMAP

################################################################################
# Configuration classes
################################################################################

# YAML valid confuguration keys.
# The main experiment configuration class is `ExecutionConfig`

config_version: str = "1.0"


@dataclass
class WindowedConfig:
    fit_on: Optional[str]
    transform_on: Optional[str]


@dataclass
class ReducerConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict]


@dataclass
class TransformConfig:
    name: str
    transform: str
    kwargs: Optional[dict] = None
    windowed: Optional[WindowedConfig] = None


@dataclass
class EstimatorConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict] = None
    num_runs: Optional[int] = 1  # Number of runs (fit/predict) for the estimator


@dataclass
class ScalerConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict] = None


@dataclass
class ExtraConfig:
    in_use_features: list
    reduce_on: str  # valid values: all, sensor, axis
    scale_on: str  # valid values: self, train


@dataclass
class ExecutionConfig:
    # control variables
    version: str
    # Datasets to use
    reducer_dataset: Optional[List[str]]
    train_dataset: List[str]
    test_dataset: List[str]
    # Transforms
    transforms: Optional[List[TransformConfig]]
    # Reducer
    reducer: Optional[ReducerConfig]
    # Scaler
    scaler: Optional[ScalerConfig]
    # Estimator
    estimators: List[EstimatorConfig]
    # Extra
    extra: ExtraConfig


################################################################################
# Transforms
################################################################################


class Identity(Transform):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def transform(self, X):
        return X


################################################################################
# Constants (Valid keys)
################################################################################

# Dictionary with the valid estimators keys to use in experiment configuration
# (under estimator.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Estimators must be a subclass of `librep.estimators.base.BaseEstimator` or implement
# the same interface (scikit-learn compatible, fit/predict methods)
estimator_cls = {
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "RandomForest": RandomForestClassifier,
}

# Dictionary with the valid reducer keys to use in experiment configuration
# (under reducer.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Reducers must be a subclass of `librep.reducers.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
reducers_cls = {"identity": Identity, 
                "umap": UMAP,
                "SimCLR":SimCLR,
                "SimCLR_linear":SimCLR_linear,
                "SimCLR_full":SimCLR_full
               }

# Dictionary with the valid transforms keys to use in experiment configuration
# (under transform.transform key).
# The key is the algorithm name and the value is the class to use.
# Transforms must be a subclass of `librep.transforms.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
transforms_cls = {
    "identity": Identity,
    "fft": FFT,
 #   "spectrogram": Spectrogram
}

# Dictionary with the valid scalers keys to use in experiment configuration
# (under scaler.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Scalers must be a subclass of `librep.scalers.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
scaler_cls = {
    "identity": Identity,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}

# Dictionary with standard labels for each activity code
standard_labels_activity = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}
