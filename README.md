# README

## Tiny NeRF dataset

![image-20230608154935173](image-20230608154935173.png)

Originally Tiny NeRF dataset was provided by UC Berkely EECS before. However currently UC Berkely does NOT provide this dataset. I don't know the reason why. So instead of the UC Berkely's link, I will use the link that UC San Diego provides for a while.

```bash
# clone this repo
git clone https://github.com/howsmyanimeprofilepicture/nerf-torch
cd ./nerf-torch
# Download Tiny NeRF Dataset from the UCSD's server.
wget https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
python ./main.py

```

## References

1. Aritra Roy Gosthipaty, Ritwik Raha (2021). ["3D volumetric rendering with NeRF"](https://keras.io/examples/vision/nerf/) (Keras Tutorials)
2. ["bmild/nerf"](https://github.com/bmild/nerf) (Github Repository)
3. ["Representing Scenes as Neural Radiance Fields for View Synthesis"](https://www.matthewtancik.com/nerf) (ECCV 2020, Tancik et al)