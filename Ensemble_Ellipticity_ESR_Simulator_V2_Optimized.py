import numpy as np
from scipy import linalg
from multiprocessing import Pool

# ========== OPTIMIZED CORE FUNCTIONS ==========
# Pré-calculer les matrices constantes
SZ = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.complex128)
SZ2 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.complex128)
IZ = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.complex128)
RHO0_SIMPLE = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.complex128)

# Constants
D = 2.87e3
GAMMA_E = 28.024e3
AZZ = 2.14
SQRT2 = np.sqrt(2)
SQRT2_INV = 1 / SQRT2

def Hint_e(phase, Ox, coef_x, Oy, coef_y):
    """Interaction Hamiltonian V2 - optimized"""
    Ax = Ox * coef_x[0] / (2 * SQRT2)
    Ay = Ox * coef_x[1] / (2 * SQRT2)
    Bx = Oy * coef_y[0] / (2 * SQRT2)
    By = Oy * coef_y[1] / (2 * SQRT2)
    
    exp_neg = np.exp(-1j * phase)
    exp_pos = np.exp(1j * phase)
    
    # Build Hamiltonian efficiently
    H_int = np.array([
        [0, (Ax + By * exp_neg), 0],
        [(Ax + By * exp_pos), 0, (Ax - By * exp_pos)],
        [0, (Ax - By * exp_neg), 0]
    ], dtype=np.complex128)
    
    H_int += np.array([
        [0, Bx * exp_neg, 0],
        [Bx * exp_pos, 0, Bx * exp_pos],
        [0, Bx * exp_neg, 0]
    ], dtype=np.complex128)
    
    H_int += np.array([
        [0, -1j * Ay, 0],
        [1j * Ay, 0, -1j * Ay],
        [0, 1j * Ay, 0]
    ], dtype=np.complex128)
    
    return H_int

def H0_e(Bz, W_mw):
    """Ground State Hamiltonian - optimized"""
    return D * SZ2 + GAMMA_E * Bz * SZ - W_mw * SZ2

def MeanPop_HF_optimized(t, Bz, W_mw, phase, Ox, coef_x, Oy, coef_y, n_time_points=5):
    """Optimized MeanPop_HF with reduced time points"""
    tlist = np.linspace(0, t, n_time_points)
    
    # Build Hamiltonian
    H0 = H0_e(Bz, W_mw)
    H_mw = Hint_e(phase, Ox, coef_x, Oy, coef_y)
    H_e = H0 + H_mw
    
    # Hyperfine coupling
    Hhf = AZZ * np.kron(IZ, IZ)
    H = np.kron(H_e, np.eye(3, 3, dtype=np.complex128)) + Hhf
    
    # Pre-compute initial state
    rhoT = np.kron(RHO0_SIMPLE, (1/3) * np.eye(3, 3, dtype=np.complex128))
    Proj0 = np.kron(RHO0_SIMPLE, np.eye(3, 3, dtype=np.complex128))
    
    # Compute evolution operators
    U = np.array([linalg.expm(-1j * H * ti) for ti in tlist])
    
    # Vectorized state evolution
    states = np.array([U[i] @ rhoT @ U[i].conj().T for i in range(len(tlist))])
    
    # Compute populations
    Pop0 = np.array([np.real(np.trace(Proj0 @ states[i])) for i in range(len(tlist))])
    
    return np.mean(Pop0)

def SingleNV_spectrum_HF_optimized(Bz, phase, Ox, coef_x, Oy, coef_y, freqList, n_time_points=5):
    """Optimized single NV spectrum computation"""
    RawVals = np.array([
        MeanPop_HF_optimized(5, Bz, w, phase, Ox, coef_x, Oy, coef_y, n_time_points) 
        for w in freqList
    ])
    
    max_val = np.max(RawVals)
    if max_val > 0:
        Contrast = RawVals / max_val
    else:
        Contrast = RawVals
    
    return Contrast

def compute_single_nv_axis(args):
    """Worker function for parallel processing of single NV axis"""
    Bz, phase, Omx, coef_x, Omy, coef_y, freqList, n_time_points = args
    return SingleNV_spectrum_HF_optimized(Bz, phase, Omx, coef_x, Omy, coef_y, freqList, n_time_points)

# ========== TRANSFORMATION FUNCTIONS ==========
def rot_x(alpha):
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca]
    ])

def rot_y(beta):
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array([
        [cb, 0, sb],
        [0, 1, 0],
        [-sb, 0, cb]
    ])

def extract_numeric_coefficients(alpha, beta):
    """Extract numeric coefficients - optimized"""
    sqrt3_inv = 1 / np.sqrt(3)
    sqrt23 = np.sqrt(2/3)
    
    TlabNV = [
        np.array([
            [sqrt3_inv, 0, sqrt23],
            [0, 1, 0],
            [-sqrt23, 0, sqrt3_inv]
        ]),
        np.array([
            [sqrt3_inv, 0, -sqrt23],
            [0, 1, 0],
            [sqrt23, 0, sqrt3_inv]
        ]),
        np.array([
            [1, 0, 0],
            [0, sqrt3_inv, -sqrt23],
            [0, sqrt23, sqrt3_inv]
        ]),
        np.array([
            [1, 0, 0],
            [0, sqrt3_inv, sqrt23],
            [0, -sqrt23, sqrt3_inv]
        ])
    ]
    
    # Basis vectors
    vx = SQRT2_INV * np.array([1, -1, 0])
    vy = SQRT2_INV * np.array([1, 1, 0])
    
    Rtot = rot_x(alpha) @ rot_y(beta)
    R_T = Rtot.T
    
    results = []
    for i, T in enumerate(TlabNV, start=1):
        coeff_x = T @ R_T @ vx
        coeff_y = T @ R_T @ vy
        
        results.append({
            "NV": i,
            "Ox*cos(a)": coeff_x,
            "Oy*sin(b)": coeff_y
        })
    
    return results

def transform_field_toNV(B, alpha=0, beta=0):
    """Transform magnetic field to NV basis - optimized"""
    sqrt3_inv = 1 / np.sqrt(3)
    sqrt23 = np.sqrt(2/3)
    
    Bx_NV = SQRT2_INV * (B[0] + B[1])
    By_NV = SQRT2_INV * (-B[0] + B[1])
    B_column = np.array([[Bx_NV], [By_NV], [B[2]]])
    
    # Rotation matrix
    Rtot = rot_x(alpha) @ rot_y(beta)
    
    # NV transformation matrices
    TlabNV = [
        np.array([[sqrt3_inv, 0, sqrt23],
                  [0, 1, 0],
                  [-sqrt23, 0, sqrt3_inv]]),
        np.array([[sqrt3_inv, 0, -sqrt23],
                  [0, 1, 0],
                  [sqrt23, 0, sqrt3_inv]]),
        np.array([[1, 0, 0],
                  [0, sqrt3_inv, -sqrt23],
                  [0, sqrt23, sqrt3_inv]]),
        np.array([[1, 0, 0],
                  [0, sqrt3_inv, sqrt23],
                  [0, -sqrt23, sqrt3_inv]])
    ]
    
    # Apply rotation
    TlabNV_rotated = [T @ Rtot.T for T in TlabNV]
    
    # Transform B field for all NV axes
    B_in_NV_basis = {}
    for i, T in enumerate(TlabNV_rotated, 1):
        B_transformed = T @ B_column
        key = f'NV_{i}'
        B_in_NV_basis[key] = [float(B_transformed[j, 0]) for j in range(3)]
    
    return B_in_NV_basis

# ========== ENSEMBLE SPECTRUM FUNCTIONS ==========
def Ensemble_Spectrum_HF(B, MW_field, MW_phase, freqList=None, tilt_x=0, tilt_y=0,
                         use_multiprocessing=True, n_time_points=5):
    """
    Optimized Ensemble Spectrum V2 computation with:
    - Multiprocessing for parallel NV axis computation
    - Reduced time points (default: 5 instead of 10)
    - Vectorized operations
    
    Parameters:
    - use_multiprocessing: Enable/disable parallel processing (default: True)
    - n_time_points: Number of time points for evolution (default: 5, original: 10)
    """
    B_inNV = transform_field_toNV(B, tilt_x, tilt_y)
    num_coeffs = extract_numeric_coefficients(tilt_x, tilt_y)
    
    Omx = MW_field[0]
    Omy = MW_field[1]
    
    # Prepare arguments for all 4 NV axes
    args_list = []
    for idx, nv_key in enumerate(B_inNV):
        Bz = B_inNV[nv_key][2]
        coef_x = num_coeffs[idx]['Ox*cos(a)']
        coef_y = num_coeffs[idx]['Oy*sin(b)']
        args_list.append((Bz, MW_phase, Omx, coef_x, Omy, coef_y, freqList, n_time_points))
    
    if use_multiprocessing and len(args_list) > 1:
        # Parallel computation of 4 NV axes
        with Pool(processes=4) as pool:
            results = pool.map(compute_single_nv_axis, args_list)
        
        # Sum all contributions
        EnsCont = np.sum(results, axis=0) / 4
    else:
        # Sequential computation (fallback)
        EnsCont = np.zeros(len(freqList))
        for args in args_list:
            Contrast = compute_single_nv_axis(args)
            EnsCont += Contrast
        EnsCont /= 4
    
    return EnsCont

# For backward compatibility
def Ensemble_Spectrum(B, MW_field, MW_phase, freqList=None, tilt_x=0, tilt_y=0):
    """Wrapper for backward compatibility - calls optimized HF version"""
    return Ensemble_Spectrum_HF(B, MW_field, MW_phase, freqList, tilt_x, tilt_y,
                                use_multiprocessing=True, n_time_points=5)
