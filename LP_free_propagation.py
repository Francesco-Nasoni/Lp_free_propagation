import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import sys

from source.propagation import free_propagation_asm_hankel, verify_resolution
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# --------------------------------------- PARAMETERS ----------------------------------------------
# -------------------------------------------------------------------------------------------------
# NOTE: all the length are measured in units of fiber radius

SUPERVISION_MODE = False
N_THREADS = 14

# --- Various Parameters ---
FIBER_V = 5.8  # V number of the fiber, must be the same used in the filed generation
DIST_FROM_FIBER = np.arange(0, 2, 1)  # Array of propagation distances from fiber end
RZ_FACTOR = 0.8  # Scaling factor of the calculated space for ASM propagation
LAMBDA = 0.0443

# --- Grid stuff ---
AXIS_SIZE = 1.3
DX_PROPAGATED_FIELD = 1/24  # Minimal spatial resolution required for ASM propagation
OVERSAMPLING_X = 4  # Required oversampling factor for dx w.r.t. mimimum Nyquist resolution (half of characteristich length)
OVERSAMPLING_Z = 4  # Required oversampling factor for dz w.r.t. mimimum Nyquist resolution

# --- Visualization stuff ---
CMAP = plt.get_cmap("gnuplot2", 20)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# --- Create output folder ---
output_folder = Path("output")
output_folder.mkdir(exist_ok=True)

# --- Fiber radius ---
radius = 1.0

# --- Some stuff on angles ---
NA = LAMBDA * FIBER_V / (2 * np.pi * radius)

# --- Grid ---
axis_ext = AXIS_SIZE * radius

K_MAX_FACTOR = 2

# --- Load guided modes from files ---
guided_modes = np.load("guided_modes.npy", allow_pickle="True")
df_coeff_fib_prop = pd.read_csv(
    "propagate_field_coeff.csv",
    dtype={0: float, 1: float, 2: float},
    converters={3: complex, 4: complex, 5: complex, 6: complex},
)
df_coeff_fib_prop.set_index(["l", "m"], inplace=True)


# -------------------------------------------------------------------------------------------------
# --- PROPAGATE THE FIELD USING ASM TO z=DIST_FROM_FIBER ---
# -------------------------------------------------------------------------------------------------


def process_propagation(Z_dist):
    """
    Process propagation for a single distance.
    The function is designed to be passed to each worker in a parallel execution context.
    """
    E_propagated_x, E_propagated_y, prop_axis_ext = free_propagation_asm_hankel(
        guided_modes,
        df_coeff_fib_prop,
        Z_dist,
        NA,
        RZ_FACTOR,
        DX_PROPAGATED_FIELD,
        K_MAX_FACTOR,
        FIBER_V,
        1.3 * axis_ext,
        min_point_per_period=10,
        radius=radius,
        lambda_0=LAMBDA,
    )

    intensity = np.abs(E_propagated_x) ** 2 + np.abs(E_propagated_y) ** 2

    if not SUPERVISION_MODE:
        np.savez(
            output_folder / f"propagated_field_z{Z_dist}.npz",
            intensity=intensity,
            dist_from_fiber=Z_dist,
            axis_ext=prop_axis_ext,
            fiber_radius=radius,
            lambda_0=LAMBDA,
        )

    return intensity, prop_axis_ext


if SUPERVISION_MODE:

    # References for resolution
    print("*" * 50)
    print(
        f"Characteristic length on the x,y directions: Lx = {LAMBDA/NA:.3f}\t currently using: δx = {DX_PROPAGATED_FIELD:.3f}"
    )
    print(
        f"Characteristic length on the z direction   : Lz = {2*LAMBDA/(NA**2):.3f}\t currently using: δz = {DIST_FROM_FIBER[1]- DIST_FROM_FIBER[0]:.3f}"
    )
    print("*" * 50, "\n")

    # Sequential execution for visualization in supervision mode
    for Z_dist in tqdm(DIST_FROM_FIBER, desc="Propagating field"):

        I_propagated, prop_axis_ext = process_propagation(Z_dist)

        if Z_dist == DIST_FROM_FIBER[0]:
            plt.figure(figsize=(10, 8))
            im = plt.imshow(
                I_propagated,
                extent=[-prop_axis_ext, prop_axis_ext, -prop_axis_ext, prop_axis_ext],
                cmap=CMAP,
                origin="lower",
            )
            plt.colorbar(im, label="Intensity")
            plt.xlabel("x")
            plt.ylabel("y")
            title = plt.title(f"Propagated Field Intensity at z={Z_dist}")
            plt.tight_layout()
            plt.ion()
            plt.show()
        else:
            im.set_data(I_propagated)
            im.set_extent(
                [-prop_axis_ext, prop_axis_ext, -prop_axis_ext, prop_axis_ext]
            )
            title.set_text(f"Propagated Field Intensity at z={Z_dist}")
            im.autoscale()
            plt.draw()
            plt.pause(0.05)
else:

    try:
        dz = DIST_FROM_FIBER[1] - DIST_FROM_FIBER[0]
        verify_resolution(
            DX_PROPAGATED_FIELD,
            dz,
            LAMBDA,
            NA,
            OVERSAMPLING_X,
            OVERSAMPLING_Z,
        )
    except Exception as e:
        print(f"ERROR:\n{e}")
        sys.exit(1)

    # Parallel execution for file saving
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = {
            executor.submit(process_propagation, Z_dist): Z_dist
            for Z_dist in DIST_FROM_FIBER
        }

        for future in tqdm(
            as_completed(futures), total=len(DIST_FROM_FIBER), desc="Propagating field"
        ):
            future.result()
