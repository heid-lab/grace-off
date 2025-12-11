<p align="center">
  <img width="128" height="128" alt="grace-off icon"
       src="https://github.com/user-attachments/assets/8408bc39-d666-4682-9ed8-0996b057b9e4" />
  <br/>
  <sub><b>grace-off</b> — ML potentials for liquids</sub>
</p>


| Model | Name   | Content                         | # elements | # Structures           |
|:-----:|:-------|:--------------------------------|------------|------------------------|
| A     | a_wpS  | water + pubSolv                 |    10 (H,C, N, O, F, P, S, Cl, Br, I)      | 14,934                 |
| B     | b_off  | SPICE (w/o ions, charged)       |    10 (H,C, N, O, F, P, S, Cl, Br, I)      | —                      |
| C     | c_all  | all                             |    17 (H, Li, B, C, N, O, F, Na, Mg, Si, P, S, Cl, K, Ca, Br, I)     |1,998,126 / 2,008,126 |

#### Different model sizes for the **1LAYER** and **2LAYER** models

| # Layers | Model  | # Parameters | l_max   | max_order | n_rad_max   | n_mlp_dens |
|---------:|--------|--------------:|---------|-----------|-------------|-------------|
| 1        | small  |       58 072  | 3       | 3         | 20          | 10          |
| 1        | medium |      512 128  | 4       | 4         | 32          | 12          |
| 1        | large  |    1 026 592  | 4       | 4         | 48          | 16          |
| 2        | small  |      235 976  | [3, 2]  | 3         | [20, 32]    | 8           |
| 2        | medium |      901 300  | [3, 3]  | 4         | [32, 42]    | 10          |
| 2        | large  |    1 626 848  | [4, 3]  | 4         | [32, 48]    | 12          |

> These are the parameters (taken from [here](https://gracemaker.readthedocs.io/en/latest/gracemaker/presets/#grace_2layer))
> - **lmax**,  maximum per-layer l-character for constructing product functions
> - **max_order**, maximum product order for B-functions
> - **n_rad_max**, per-layer max number of radial functions
> - **n_mlp_dens**, number of non-linear readout densities


