import logging
from pprint import pformat
from typing import Dict, Any
from pathlib import Path

import rdkit
from torch_geometric.datasets import MoleculeNet

logger = logging.getLogger(__name__)


class MoleculeNetDataSet:
    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        
    def load(self) -> None:
        pass
    
    @staticmethod
    def _load_from_package() -> MoleculeNet:
        data = MoleculeNet(root=".", name="ESOL")
        
        dict_describe: Dict[str, Any] = {
            "type": type(data),
            "features": data.num_features,
            "target": data.num_classes,
            "length": data.len(),
            "first_entry": data[0],
            "first_graph_n_nodes": data[0].num_nodes,
            "first_graph_n_edges": data[0].num_edges,
        }
        logger.info("Loaded data with the following stats:\n"
                    f"{pformat(dict_describe)}")
        
        return data
        

def geometric_deep_learning_pipeline(path_molecule_net_data: Path) -> None:
    # Load raw data
    dataset = MoleculeNetDataSet(filepath=path_molecule_net_data)
    data = dataset._load_from_package()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    PATH_MOLECULE_NET_DATA: Path = Path("data") / "01_raw" / "molecule_net_data.json"
    
    geometric_deep_learning_pipeline(
        path_molecule_net_data=PATH_MOLECULE_NET_DATA
    )
