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
                 nanomaterial_size_angstrom=50, # 10, 50, 100, 1000
                 n_presubsample=4096, n_postsubsample=512,
                 min_2_theta = 0, 
                 max_2_theta = 180,
                 wavesource='CuKa',
                 horizontal_noise_range=(1e-2, 1.1e-2), # (1e-3, 1.1e-3)
                 vertical_noise=1e-3,
                 pdf=False, normalized_pdf=False,
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
        self.pdf = pdf
        self.normalized_pdf = normalized_pdf
        assert self.xrd_filter in ['gaussian', 'sinc', 'both'], "invalid filter requested"

        self.wavelength = WAVELENGTHS[wavesource]
        self.nanomaterial_size = nanomaterial_size_angstrom
        self.n_presubsample = n_presubsample
        self.n_postsubsample = n_postsubsample

        if self.xrd_filter == 'sinc' or self.xrd_filter == 'both':
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
            
            self.sinc_filt = self.nanomaterial_size * (np.sinc(self.nanomaterial_size * self.Qs_shifted / np.pi) ** 2)
            # sinc filter is symmetric, so we can just use the first half
        else:
            raise ValueError("Gaussian filter is deprecated. Use sinc filter instead.")

        self.horizontal_noise_range=horizontal_noise_range
        self.vertical_noise=vertical_noise

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

        # smooth XRDs
        for curr_data_dict in tqdm(self.cached_data):
            curr_xrd = curr_data_dict[self.prop]
            curr_xrd = curr_xrd.reshape((self.n_presubsample,))
            curr_data_dict['rawXRD'] = self.sample(curr_xrd.numpy()) # need to downsample first
            # have sinc with gaussian filter & sinc w/out gaussian filter
            if self.pdf:
                sample_interval = self.n_presubsample // self.n_postsubsample
                Qs_sampled = torch.tensor(self.Qs[::sample_interval])
                sinc_only_postsubsample = self.augment_xrdStrip(curr_xrd, return_both=False) # take sinc filtered xrd
                rs, the_pdf = self.overall_pdf(Qs=Qs_sampled, signal=sinc_only_postsubsample, num_samples=self.n_postsubsample)
                curr_data_dict[self.prop] = the_pdf
                curr_data_dict['sincOnly'] = sinc_only_postsubsample
                curr_data_dict['sincOnlyPresubsample'] = None
                curr_data_dict['xrdPresubsample'] = curr_xrd
            else:
                curr_xrd, sinc_only_xrd, curr_xrd_presubsample, sinc_only_xrd_presubsample = self.augment_xrdStrip(curr_xrd, return_both=True)
                curr_data_dict[self.prop] = curr_xrd
                curr_data_dict['sincOnly'] = sinc_only_xrd
                curr_data_dict['sincOnlyPresubsample'] = sinc_only_xrd_presubsample
                curr_data_dict['xrdPresubsample'] = curr_xrd_presubsample

    def overall_pdf(self, Qs, signal, r_min=0, r_max=30, num_samples=512):
        assert Qs.shape == signal.shape
        signal = signal / torch.mean(signal)
        rs = torch.linspace(r_min, r_max, num_samples)
        rs_orig = rs
        the_pdf = list()
        assert torch.isclose(torch.mean(signal), torch.tensor(1.0).to(dtype=signal.dtype))
        delta_Q = torch.tensor((Qs[-1] - Qs[0]) / (Qs.shape[0] - 1), dtype=signal.dtype)
        assert torch.isclose(delta_Q, torch.tensor(Qs[1] - Qs[0], dtype=signal.dtype))
        Qs = Qs.reshape(-1, 1).expand(-1, num_samples)
        signal = signal.reshape(-1, 1).expand(-1, num_samples)
        rs = rs.reshape(1, -1).expand(Qs.shape[0], -1)
        assert Qs.shape == signal.shape == rs.shape
        the_pdf = torch.sum(2 / np.pi * Qs * (signal - 1) * torch.sin(Qs * rs) * delta_Q, 0)
        assert the_pdf.shape == (num_samples,)
        if self.normalized_pdf:
            the_pdf /= torch.max(torch.abs(the_pdf))
        return rs_orig, the_pdf

    def sample(self, x):
        step_size = int(np.ceil(len(x) / self.n_postsubsample))
        x_subsample = [np.max(x[i:i+step_size]) for i in range(0, len(x), step_size)]
        return np.array(x_subsample)

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        # scaler is set in DataModule set stage
        prop = data_dict[self.prop]#self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']
        
        if "xrd" in data_dict.keys():
            assert self.n_postsubsample == 512
            dim = 512
            prop = prop.view(dim, -1)
        else:
            dim = 1

        # store raw sinc for plotting
        if self.xrd_filter == 'both':
            raw_sinc = data_dict['sincOnly']
            assert self.n_postsubsample == 512
            raw_sinc = raw_sinc.view(self.n_postsubsample, -1)
            raw_sinc_presubsample = data_dict['sincOnlyPresubsample']
            xrd_presubsample = data_dict['xrdPresubsample']
        else:
            raw_sinc = data_dict['sincOnly'].view(self.n_postsubsample, 1)
            raw_sinc_presubsample = None
            xrd_presubsample = None

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
            mpid=data_dict['mp_id'],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop,
            raw_sinc=raw_sinc,
            raw_sinc_presubsample=raw_sinc_presubsample,
            xrd_presubsample=xrd_presubsample,
            raw_xrd=torch.tensor(data_dict['rawXRD'])
        )
        return data

    def sinc_filter(self, x):
        filtered = np.convolve(x, self.sinc_filt, mode='same')
        return filtered
    
    def gaussian_filter(self, x):
        filtered = gaussian_filter1d(x,
                    sigma=np.random.uniform(
                        low=self.n_presubsample * self.horizontal_noise_range[0], 
                        high=self.n_presubsample * self.horizontal_noise_range[1]
                    ), 
                    mode='constant', cval=0)    
        return filtered

    def augment_xrdStrip(self, curr_xrdStrip, return_both=False):
        """
        Input:
        -> curr_xrdStrip: XRD pattern of shape (self.n_presubsample,)
        -> return_both: if True, return (bothFiltered, rawSincFiltered), only valid if self.xrd_filter == 'both';
            if False, return based on self.xrd_filter
        Output:
        -> if return_both=False, 
            returns curr_xrdStrip augmented by peak broadening (sinc and/or gaussian) & vertical Gaussian perturbations;
            with shape (self.n_postsubsample,); in range [0, 1]
        -> if return_both=True,
            returns (bothFiltered, rawSincFiltered); where bothFiltered has both sinc filter & gaussian filter,
            rawSincFiltered has only sinc filter
        """
        if self.pdf:
            assert self.xrd_filter == 'sinc'
        xrd = curr_xrdStrip.numpy()
        assert xrd.shape == (self.n_presubsample,)
        # Peak broadening
        if self.xrd_filter == 'both':
            sinc_filtered = self.sinc_filter(xrd)
            filtered = self.gaussian_filter(sinc_filtered)
            sinc_only_presubsample = torch.from_numpy(sinc_filtered)
            assert filtered.shape == xrd.shape
            assert not self.pdf
        elif self.xrd_filter == 'sinc':
            filtered = self.sinc_filter(xrd)
            assert filtered.shape == xrd.shape
        elif self.xrd_filter == 'gaussian':
            filtered = self.gaussian_filter(xrd)
            assert filtered.shape == xrd.shape
        else:
            raise ValueError("Invalid filter requested")
                
        assert filtered.shape == curr_xrdStrip.shape

        # presubsamples
        filtered_presubsample = torch.from_numpy(filtered)

        # postsubsampling
        filtered_postsubsampled = self.post_process_filtered_xrd(filtered)

        if return_both: # want to return double filtered & sinc-only filtered
            assert not self.pdf
            assert self.xrd_filter == 'both'
            assert sinc_filtered.shape == curr_xrdStrip.shape
            # postsubsampling
            sinc_only_postsubsample = self.post_process_filtered_xrd(sinc_filtered)
            assert filtered_presubsample.shape == sinc_only_presubsample.shape == (self.n_presubsample,)
            assert filtered_postsubsampled.shape == sinc_only_postsubsample.shape == (self.n_postsubsample,)
            return filtered_postsubsampled, sinc_only_postsubsample, filtered_presubsample, sinc_only_presubsample
        return filtered_postsubsampled
    
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
