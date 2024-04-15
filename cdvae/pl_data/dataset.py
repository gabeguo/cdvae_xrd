import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset

from scipy.ndimage import gaussian_filter1d
import numpy as np

from torch_geometric.data import Data
import os

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)
from pymatgen.analysis.diffraction.xrd import WAVELENGTHS

from tqdm import tqdm

class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, xrd_filter : ValueNode,
                 nanomaterial_size_min_angstrom=10,
                 nanomaterial_size_max_angstrom=1000,
                 n_presubsample=4096, n_postsubsample=512,
                 min_2_theta = 0, 
                 max_2_theta = 180,
                 wavesource='CuKa',
                 vertical_noise=1e-2,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_pickle(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.xrd_filter = xrd_filter
        assert self.xrd_filter in ['sinc'], "invalid filter requested"

        assert self.prop == 'xrd'

        self.wavelength = WAVELENGTHS[wavesource]
        self.nanomaterial_size_min = nanomaterial_size_min_angstrom
        self.nanomaterial_size_max = nanomaterial_size_max_angstrom
        self.n_presubsample = n_presubsample
        self.n_postsubsample = n_postsubsample

        if self.xrd_filter == 'sinc':
            # compute Q range
            min_theta = min_2_theta / 2
            max_theta = max_2_theta / 2
            Q_min = 4 * np.pi * np.sin(np.radians(min_theta)) / self.wavelength
            Q_max = 4 * np.pi * np.sin(np.radians(max_theta)) / self.wavelength

            # phase shift for sinc filter = half of the signed Q range
            phase_shift = (Q_max - Q_min) / 2

            # compute Qs
            self.Qs = np.linspace(Q_min, Q_max, self.n_presubsample)
            self.Qs_shifted = self.Qs - phase_shift            
            
        else:
            raise ValueError("Other filters are deprecated. Use sinc filter instead.")

        self.vertical_noise = vertical_noise

        self.cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop])

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def sample(self, x):
        step_size = int(np.ceil(len(x) / self.n_postsubsample))
        x_subsample = [np.max(x[i:i+step_size]) for i in range(0, len(x), step_size)]
        return np.array(x_subsample)

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        assert self.prop == 'xrd'
        raw_xrd = data_dict[self.prop]
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']
        
        assert raw_xrd.shape == (self.n_presubsample,)
        augmented_xrd, curr_nanomaterial_size = self.augment_xrdStrip(raw_xrd)
        assert augmented_xrd.shape == (self.n_postsubsample,)

        if "xrd" in data_dict.keys():
            assert self.n_postsubsample == 512
            dim = 512
            augmented_xrd = augmented_xrd.view(dim, -1)
            assert augmented_xrd.shape == (512, 1)

            plain_xrd = self.sample(raw_xrd.numpy())
            plain_xrd = torch.tensor(plain_xrd).view(dim, -1)
            assert plain_xrd.shape == (512, 1)
        else:
            raise ValueError('should have xrd')
            dim = 1

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            spacegroup=data_dict['spacegroup.number'],
            pretty_formula=data_dict['pretty_formula'],
            cif=data_dict['cif'],
            mpid=data_dict['mp_id'],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=augmented_xrd.to(dtype=torch.float32),
            raw_xrd=plain_xrd.to(dtype=torch.float32),
            nanomaterial_size=curr_nanomaterial_size
        )
        return data

    def sinc_filter(self, x):
        curr_nanomaterial_size = np.random.uniform(low=self.nanomaterial_size_min,
                                                   high=self.nanomaterial_size_max)
        curr_sinc_filt = curr_nanomaterial_size * np.sinc((np.pi * curr_nanomaterial_size * self.Qs_shifted)/np.pi)
        filtered = np.convolve(x, curr_sinc_filt, mode='same')
        return filtered, curr_nanomaterial_size
    
    def augment_xrdStrip(self, curr_xrdStrip):
        """
        Input:
        -> curr_xrdStrip: XRD pattern of shape (self.n_presubsample,)
        Output:
        -> returns curr_xrdStrip augmented by peak broadening (sinc) & vertical Gaussian perturbations;
            with shape (self.n_postsubsample,); in range [0, 1]
        """
        xrd = curr_xrdStrip.numpy()
        assert xrd.shape == (self.n_presubsample,)
        # Peak broadening
        if self.xrd_filter == 'sinc':
            filtered, curr_nanomaterial_size = self.sinc_filter(xrd)
            assert filtered.shape == xrd.shape
        else:
            raise ValueError("Invalid filter requested")
                
        assert filtered.shape == curr_xrdStrip.shape

        # presubsamples
        filtered_presubsampled = torch.from_numpy(filtered)

        # postsubsampling
        filtered_postsubsampled = self.post_process_filtered_xrd(filtered)

        return filtered_postsubsampled, curr_nanomaterial_size
    
    def post_process_filtered_xrd(self, filtered):
        # scale
        filtered = filtered / np.max(filtered)
        filtered = np.maximum(filtered, np.zeros_like(filtered))
        # sample it
        assert filtered.shape == (self.n_presubsample,)
        filtered = self.sample(filtered)
        # convert to torch
        filtered = torch.from_numpy(filtered)
        assert filtered.shape == (self.n_postsubsample,)
        # Perturbation
        perturbed = filtered + torch.normal(mean=0, std=self.vertical_noise, size=filtered.size())
        perturbed = torch.maximum(perturbed, torch.zeros_like(perturbed))
        perturbed = torch.minimum(perturbed, torch.ones_like(perturbed)) # band-pass filter
        return perturbed

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"
        
class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from cdvae.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    # main()
    Dataset = CrystDataset(
        name="mp_20",
        path="/home/tsaidi/Research/cdvae_xrd/data/mp_20/test.csv",
        prop="xrd",
        niggli=True,
        primitive=True,
        graph_method="crystalnn",
        preprocess_workers=30,
        lattice_scale_method="scale_length",
        xrd_filter="both",
    )
