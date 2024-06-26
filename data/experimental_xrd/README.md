Data sourced from [IUCr](https://www.iucr.org/resources/data/databases). All rights belong to them and the original authors.

We have the following files.
```
av5088sup4.rtv.combined.cif
br1322Isup2.rtv.combined.cif
br1340Isup2.rtv.combined.cif
ck5030Vsup6.rtv.combined.cif
gw5052Mg2Sn_100K_LTsup23.rtv.combined.cif
gw5052Mg2Si_100K_LTsup2.rtv.combined.cif
ks5409BTsup2.rtv.combined.cif
sh0123Xraysup5.rtv.combined.cif
sq1033Isup2.rtv.combined.cif
sq3214Isup2.rtv.combined.cif
iz1026Isup2.rtv.combined.cif
wh5012phaseIIsup2.rtv.combined.cif
wh5012phaseIIIsup3.rtv.combined.cif
wm2446Isup2.rtv.combined.cif
wn6225Isup2.rtv.combined.cif
```

Unnecessary to do, but we ran this (with the appropriate changes to the filepaths in ```read_real_xrd.py```), to process the files:
```
cd ../../process_real_xrds
python read_real_xrd.py
```

That gives us a subfolder ```cif_files``` and a file ```test.csv``` in this directory.
