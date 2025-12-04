from .build import build_dataset_from_cfg
import datasets.PCNDataset
import datasets.EPNDataset  # Make sure this file registers EPN3D
import datasets.ScanNetDataset
import datasets.EPNCompleteDataset


from .shapenet_dataset import ShapeNetDataset
from utils.registry import Registry

# Initialize the DATASETS registry
DATASETS = Registry("dataset")

# Register ShapeNetDataset in the DATASETS registry
DATASETS.register_module(name="ShapeNetDataset", module=ShapeNetDataset)

from datasets.PCNCompleteDataset import PCNCompleteDataset
DATASETS.register_module(name="PCNCompleteDataset", module=PCNCompleteDataset)


# Register EPN3D (alias for EPNDataset) explicitly
from datasets.EPNDataset import EPNDataset
DATASETS.register_module(name="EPN3D", module=EPNDataset)

def build_dataset_from_cfg(config, default_args=None):
    return DATASETS.build(config, default_args=default_args)
