from pymatgen.core.structure import Structure
import os
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

root_dir = '/home/gabeguo/experimental_cif'
all_filenames = list()
for filename in tqdm(os.listdir(root_dir)):
    try:
        structure = Structure.from_file(os.path.join(root_dir, filename))
        if len(structure.sites) <= 20:
            all_filenames.append(filename)
            #print(filename)
    except Exception as e:
        pass
        #print(e)
print(sorted(all_filenames))