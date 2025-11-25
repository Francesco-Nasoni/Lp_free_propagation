import numpy as np
import pandas as pd

from source.LP_projection_functions import (
    get_guided_modes,
    get_LP_modes_projection_coefficients,
    get_tilted_beam_from_incidence,
)

from source.propagation import (
    fiber_propagation,
)

# --------------------------------------- PARAMETERS ----------------------------------------------
# -------------------------------------------------------------------------------------------------
# NOTE: all the length are measured in units of fiber radius

# --- Various Parameters ---
FIBER_V = 5.8
MODES_TO_TEST = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)]
FIBER_N1 = 1
FIBER_LENGTH = 1.e4
DIST_FROM_FIBER = 800

# --- Injected field parameters ---
LAMBDA = 0.0443                 # Wavelength of the injected beam
DIST_TO_WAIST = 0               # Distance from the beam waist to the fiber input plane
W0_X = 0.6                      # Beam waist size along the x-axis
W0_Y = 0.7                       # Beam waist size along the y-axis
X0 = -0.2                        # x-coordinate of the beam's incidence point on the fiber input plane
Y0 = -0.1                        # y-coordinate of the beam's incidence point on the fiber input plane
ROLL_ANGLE = -0 * np.pi / 180    # Roll angle of the beam (rotation about the z-axis, in radians)
PITCH_ANGLE = 0 * np.pi / 180   # Pitch angle of the beam (tilt in the x-z plane, in radians)
YAW_ANGLE = 0 * np.pi / 180     # Yaw angle of the beam (tilt in the y-z plane, in radians)
POLARIZATION_ANGLE = 0    # Polarization angle of the beam (angle of the electric field vector, in radians)

# --- Grid stuff ---
AXIS_SIZE = 1.3
GRID_SIZE = 1000

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# --- Fiber radius ---
radius = 1.0

# --- Some stuff on angles ---
NA = LAMBDA * FIBER_V / (2 * np.pi * radius)

# --- Grid ---
axis_ext = AXIS_SIZE * radius
x = np.linspace(-axis_ext, axis_ext, GRID_SIZE)
y = np.linspace(-axis_ext, axis_ext, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# --- Ploar coordinates ---
R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)

# --- Differential Area Element ---
dA = (axis_ext * 2 / GRID_SIZE) ** 2

# --- DEFINE THE INPUT ELECTRIC FIELD AS A TILTED GAUSSIAN BEAM ---
E_input = get_tilted_beam_from_incidence(
    X,
    Y,
    z_plane=0,
    x_incidence=X0,
    y_incidence=Y0,
    dist_to_waist=DIST_TO_WAIST,
    euler_alpha=ROLL_ANGLE,
    euler_beta=PITCH_ANGLE,
    euler_gamma=YAW_ANGLE,
    dA=dA,
    w0_x=W0_X,
    w0_y=W0_Y,
    wavelength=LAMBDA,
    polarization_angle=POLARIZATION_ANGLE,
)

# --- COMPUTE THE GUIDED MODES AND THEIR PROJECTION COEFFICIENTS ON THE INPUT FIELD ---
guided_modes = []
coefficients = []
for l, m in MODES_TO_TEST:

    mode = get_guided_modes(l, m, FIBER_V, radius, R, PHI, dA)

    if mode is not None:
        guided_modes.append(mode)
        coefficients_res = get_LP_modes_projection_coefficients(E_input, mode, dA)
        coefficients.append(coefficients_res)


# create a pd dataframe, and make (l,m) the index
df_coeff = pd.DataFrame(coefficients)
df_coeff.set_index(["l", "m"], inplace=True)

# --- TERMINAL OUTPUT ---
print("\n", "SQUARED MODULUS OF COEFFICIENTS", "\n" + "*" * 70)
print(
    (df_coeff.iloc[:, 1:]).to_string(
        float_format=lambda x: f"{x:.2f}", justify="center", col_space=10
    )
)
print("*" * 70 + "\n")


# --- CALCULATE THE COEFFICIENTS AFTER THE PROPAGATION INSIDE THE FIBER ---
df_coeff_fib_prop = fiber_propagation(
    df_coeff,
    n1=FIBER_N1,
    a=radius,
    lam=LAMBDA,
    z_fiber=FIBER_LENGTH,
)

df_coeff_fib_prop.to_csv("propagate_field_coeff.csv", index="True")
np.save("guided_modes.npy", guided_modes)