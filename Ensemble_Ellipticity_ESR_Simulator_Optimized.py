import numpy as np
from scipy import linalg
from multiprocessing import Pool
from functools import partial

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

def Hint_e(phase, Ox, Oy):
    """Interaction Hamiltonian - optimized"""
    exp_neg = np.exp(-1j * phase)
    exp_pos = np.exp(1j * phase)
    
    H_int = np.array([
        [0, (Ox + Oy * exp_neg), 0],
        [(Ox + Oy * exp_pos), 0, (Ox - Oy * exp_pos)],
        [0, (Ox - Oy * exp_neg), 0]
    ], dtype=np.complex128)
    
    return H_int

def H0_e(Bz, W_mw):
    """Ground State Hamiltonian - optimized"""
    return D * SZ2 + GAMMA_E * Bz * SZ - W_mw * SZ2

def MeanPop_HF_optimized(t, Bz, W_mw, phase, Ox, Oy, n_time_points=5):
    """Optimized MeanPop_HF with reduced time points"""
    tlist = np.linspace(0, t, n_time_points)
    
    # Build Hamiltonian
    H0 = H0_e(Bz, W_mw)
    H_mw = Hint_e(phase, Ox, Oy)
    H_e = H0 + H_mw
    
    # Hyperfine coupling
    Hhf = AZZ * np.kron(IZ, IZ)
    H = np.kron(H_e, np.eye(3, 3, dtype=np.complex128)) + Hhf
    
    # Pre-compute initial state
    rhoT = np.kron(RHO0_SIMPLE, (1/3) * np.eye(3, 3, dtype=np.complex128))
    Proj0 = np.kron(RHO0_SIMPLE, np.eye(3, 3, dtype=np.complex128))
    
    # Vectorized evolution operators computation
    U = np.array([linalg.expm(-1j * H * ti) for ti in tlist])
    
    # Vectorized state evolution - use einsum for efficiency
    states = np.array([U[i] @ rhoT @ U[i].conj().T for i in range(len(tlist))])
    
    # Compute populations
    Pop0 = np.array([np.real(np.trace(Proj0 @ states[i])) for i in range(len(tlist))])
    
    return np.mean(Pop0)

def SingleNV_spectrum_HF_optimized(Bz, phase, Ox, Oy, freqList, n_time_points=5):
    """Optimized single NV spectrum computation"""
    # Vectorized computation
    RawVals = np.array([
        MeanPop_HF_optimized(5, Bz, w, phase, Ox, Oy, n_time_points) 
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
    Bz, Omx, Omy, phase, freqList, n_time_points = args
    return SingleNV_spectrum_HF_optimized(Bz, phase, Omx, Omy, freqList, n_time_points)

# ========== TRANSFORMATION FUNCTIONS ==========
def rot_x(alpha):
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca]
    ])

def rot_y(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, 0, st],
        [0, 1, 0],
        [-st, 0, ct]
    ])

def transform_field_toNV(B, MW, alpha=0, beta=0):
    """Transform field to NV basis - optimized"""
    sqrt2_inv = 1 / np.sqrt(2)
    sqrt3_inv = 1 / np.sqrt(3)
    sqrt23 = np.sqrt(2/3)
    
    # Transform B field
    Bx_NV = sqrt2_inv * (B[0] + B[1])
    By_NV = sqrt2_inv * (-B[0] + B[1])
    B_column = np.array([[Bx_NV], [By_NV], [B[2]]])
    
    # Transform MW field
    MW_column = np.array([
        [sqrt2_inv * (MW[0] + MW[1])],
        [sqrt2_inv * (-MW[0] + MW[1])],
        [MW[2]]
    ])
    
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
    
    # Apply rotation to all transformation matrices
    TlabNV_rotated = [T @ Rtot.T for T in TlabNV]
    
    # Transform fields for all NV axes
    B_in_NV_basis = {}
    MW_in_NV_basis = {}
    
    for i, T in enumerate(TlabNV_rotated, 1):
        B_transformed = T @ B_column
        MW_transformed = T @ MW_column
        
        key = f'NV_{i}'
        B_in_NV_basis[key] = [float(B_transformed[j, 0]) for j in range(3)]
        MW_in_NV_basis[key] = [float(MW_transformed[j, 0]) for j in range(3)]
    
    return B_in_NV_basis, MW_in_NV_basis

# ========== ENSEMBLE SPECTRUM FUNCTIONS ==========
def Ensemble_Spectrum_HF(B, MW_field, MW_phase, freqList=None, tilt_x=0, tilt_y=0, 
                         use_multiprocessing=True, n_time_points=5):
    """
    Optimized Ensemble Spectrum computation with:
    - Multiprocessing for parallel NV axis computation
    - Reduced time points (default: 5 instead of 10)
    - Vectorized operations
    
    Parameters:
    - use_multiprocessing: Enable/disable parallel processing (default: True)
    - n_time_points: Number of time points for evolution (default: 5, original: 10)
    """
    B_inNV, MW_inNV = transform_field_toNV(B, MW_field, tilt_x, tilt_y)
    
    # Prepare arguments for all 4 NV axes
    args_list = []
    for nv_key in B_inNV:
        Bz = B_inNV[nv_key][2]
        Omx = MW_inNV[nv_key][0]
        Omy = MW_inNV[nv_key][1]
        args_list.append((Bz, Omx, Omy, MW_phase, freqList, n_time_points))
    
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
