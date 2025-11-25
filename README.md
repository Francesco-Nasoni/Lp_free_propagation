# Optical Fiber Mode Coupling and Free-Space Propagation

This repository contains a Python simulation framework for analyzing the coupling of free-space optical beams into multimode optical fibers and their subsequent diffraction upon exiting the fiber.

The project focuses on **Linearly Polarized (LP)** modes and utilizes the **Angular Spectrum Method (ASM)** for accurate free-space propagation simulations.

## Workflow

To run the simulation, you must execute the scripts in the following order:

### 1. Generation
Run **`LP_guided_field_generator.py`** first.
* This script calculates the coupling of the input beam into the fiber modes.
* It generates the necessary intermediate files (`propagate_field_coeff.csv` and `guided_modes.npy`) required for the next step.

### 2. Propagation
Run **`LP_free_propagation.py`** second.
* This script reads the generated files and simulates the field propagating into free space from the fiber tip.

## Configuring `LP_free_propagation.py`

This script operates in two distinct modes controlled by the `SUPERVISION_MODE` variable inside the file:

* **`SUPERVISION_MODE = True`**:
    * **Visualization Mode**: The script runs sequentially and opens a window showing a real-time animation of the field intensity as it propagates.
    * *Use this to quickly check the physics and field evolution.*

* **`SUPERVISION_MODE = False`**:
    * **Computation Mode**: The script runs in parallel (multithreaded) and saves the results ($E_x$, $E_y$, Intensity) into `.npz` files in the `output/` folder.
    * *Use this for generating data for analysis.*

### Key Parameters to Adjust

You can modify the simulation by changing these variables at the top of `LP_free_propagation.py`:

* **`DIST_FROM_FIBER`**: Defines the Z-positions where the field is calculated.
    * *Format:* A numpy array, e.g., `np.arange(0, 800, 1)` (start, stop, step).

* **`MIN_DX_PROPAGATED_FIELD`**: Controls the spatial resolution of the simulation grid.
    * *Effect:* A smaller value gives a finer grid (more accurate) but increases computation time.

* **`N_THREADS`**: Sets the number of parallel worker threads.
    * *Effect:* Only used when `SUPERVISION_MODE = False`. Increase this to speed up data generation on multi-core CPUs.

* **`RZ_FACTOR`**: Radial Zoom Factor.
    * *Effect:* Controls the physical size of the calculation window. If the field expands significantly, you may need to increase this to prevent the field from hitting the simulation boundaries.

## Suggestion

It is strongly recommended to follow this procedure when running new simulations:

1.  Set `SUPERVISION_MODE = True` and adjust `DIST_FROM_FIBER` to check a few specific points (e.g., one near, one intermediate, and one far distance).
2.  Run the script to verify that the field evolution looks correct and that the calculation window (`RZ_FACTOR`) is large enough to contain the diffracting beam.
3.  Once verified, set `SUPERVISION_MODE = False` and configure the full range of `DIST_FROM_FIBER`.
4.  Run the script to perform the heavy calculations and save the field data to files.