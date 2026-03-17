# Offline Training Task Log
## default folder dashboard/backend/results_ms

`NB_398_0` - `NB_398_4`: We first evaluated GANomics on the NB dataset (n = 498). Using 398 paired profiles for training and holding out 100 for testing, we repeated the train/test split five times for robustness. 

 - Observed results: Training converged in ~30 epochs, with generator, discriminator, cycle-consistency, and feedback losses stabilizing concordantly (Fig. 2a–d). The per-sample L1 distance between real microarray and RNA-seq profiles from the same sample was used as a baseline; translations with lower distances were deemed successful. GANomics fell well below the baseline on the 100 held-out samples (Fig. 2e–f).

`NB_100_0` - `NB-300-4`: To determine the number of paired samples required for reliable translation, we varied the training set size from 10 to 400 paired profiles, randomly selected from the full neuroblastoma dataset (n = 498). The remaining samples were reserved for testing. To balance training efficiency and convergence, we used 50 epochs for larger training sets (100–398 pairs), and 500 epochs for small training sets (10–50 pairs). Each train–test split was repeated five times with random sampling to ensure robustness.

`NB_Size_10_run_0` - `Nb_Size_50_run_4`: experiments running with old version of GANomics Model; the architecture of GANomics model has been optimized for its I/O to accelerate training process.


