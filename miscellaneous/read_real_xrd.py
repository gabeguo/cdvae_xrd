import torch
import argparse
import os
import matplotlib.pyplot as plt

def main(args):
    assert os.path.exists(args.filepath)
    with open(args.filepath, 'r') as fin:
        all_lines = [x.strip() for x in fin.readlines()]
        xrd_loop_start_idx = all_lines.index('loop_')
        assert xrd_loop_start_idx >= 0
        expected_fields = ['_pd_meas_intensity_total',
                                    '_pd_proc_ls_weight',
                                    '_pd_proc_intensity_calc_bkg',
                                    '_pd_calc_intensity_total']
        for idx in range(len(expected_fields)):
            assert expected_fields[idx] == all_lines[xrd_loop_start_idx + (idx + 1)]
        start_idx = xrd_loop_start_idx + (4+1)
        end_idx = all_lines.index('', start_idx)
        assert end_idx > start_idx

        xrd_tensor = torch.zeros(args.xrd_dim)

        xrd_intensities = all_lines[start_idx:end_idx]
        for idx, xrd_info in enumerate(xrd_intensities):
            xrd_info = xrd_info.split()
            assert len(xrd_info) == len(expected_fields)
            assert xrd_info[0].format
            intensity_mean, intensity_std = xrd_info[0].split('(')
            intensity_std = float(intensity_std[:-1])
            intensity_mean = float(intensity_mean)
            
            closest_tensor_idx = int(idx / len(xrd_intensities) * args.xrd_dim)
            xrd_tensor[closest_tensor_idx] = max(xrd_tensor[closest_tensor_idx],
                                                 intensity_mean)
                                                #torch.normal(mean=intensity_mean, std=intensity_std, size=(1,)).item())
        
        print('moo')
        #print(xrd_tensor)
        xrd_tensor = (xrd_tensor - torch.min(xrd_tensor)) / (torch.max(xrd_tensor) - torch.min(xrd_tensor))
        plt.plot([_ for _ in range(args.xrd_dim)], xrd_tensor.detach().cpu().numpy())
        filename = args.filepath.split('/')[-1]
        plt.savefig(f'{filename}.png')

        # print out diffraction pattern info
        radiation_info_start = all_lines.index('_reflns_number_observed', end_idx)
        radiation_info_end = all_lines.index('', radiation_info_start)
        for idx in range(radiation_info_start, radiation_info_end):
            print(all_lines[idx])

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath',
                        type=str,
                        default='/home/gabeguo/experimental_cif/an0607ARZO20A_p_01sup3.rtv.combined.cif')
    parser.add_argument('--xrd_dim',
                        type=int,
                        default=512)
    parser.add_argument('--min_theta',
                        type=int,
                        default=0)
    parser.add_argument('--max_theta',
                        type=int,
                        default=180)
    args = parser.parse_args()
    main(args)