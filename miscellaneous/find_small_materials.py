from pymatgen.core.structure import Structure
import os
import warnings
warnings.filterwarnings("ignore")

root_dir = '/home/gabeguo/experimental_cif'
for filename in os.listdir(root_dir):
    try:
        structure = Structure.from_file(os.path.join(root_dir, filename))
        if len(structure.sites) <= 20:
            print(filename)
    except Exception as e:
        pass
        #print(e)