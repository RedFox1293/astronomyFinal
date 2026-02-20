import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from dataclasses import field as dc_field
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MirrorSurface:
    R: float          # Radius of curvature (signed)
    K: float          # Conic constant
    label: str = ""
    @property
    def c(self) -> float:
        if np.isinf(self.R) or self.R == 0:
            return 0.0
        return 1.0 / self.R
    @property
    def f(self) -> float:
        return -self.R / 2.0
    def sag(self, r: np.ndarray) -> np.ndarray:
        c = self.c
        if c == 0:
            return np.zeros_like(r)
        arg = 1.0 - (1.0 + self.K) * c**2 * r**2
        arg = np.maximum(arg, 0.0)
        return c * r**2 / (1.0 + np.sqrt(arg))
@dataclass
class Detector: #For realistic gap in M2, arbitrary sizes/shroud
    size: float = 50.0          # Physical detector side length (mm)
    pixel_pitch: float = 0.01   # Pixel pitch (mm), default 10 um
    offset: Tuple[float, float] = (0.0, 0.0)  # (x, y) offset from optical axis
    shroud_radius: float = 40.0   # Outer radius of cylindrical shroud (mm)
    shroud_length: float = 80.0   # Axial length of shroud (mm)
    shroud_wall: float = 2.0      # Wall thickness of shroud (mm)
    @property
    def n_pixels(self) -> int:
        return int(self.size / self.pixel_pitch)
@dataclass
class ThreeMirrorSystem:
    M1: MirrorSurface
    M2: MirrorSurface
    M3: MirrorSurface
    d1: float          # Separation M1 -> M2 (signed)
    d2: float          # Separation M2 -> M3 (signed)
    D1: float          # Primary mirror diameter
    field_angle: float = 0.5  # Half-field angle in degrees
    detector: Detector = dc_field(default_factory=Detector)
    @property
    def h1(self) -> float:
        return self.D1 / 2.0
    @property
    def f1(self) -> float:
        return self.M1.f
    @property
    def F1(self) -> float:
        return abs(self.f1) / self.D1
@dataclass
class ParaxialRayState: #State of a paraxial ray
    h: float
    u: float
    u_prime: float
    n: float
    n_prime: float
    A: float
    delta_un: float

def paraxial_reflect(h: float, u: float, R: float, n: float) -> Tuple[float, float]:
    n_prime = -n
    u_prime = -u - 2.0 * h / R
    return u_prime, n_prime

def paraxial_transfer(h: float, u_prime: float, d: float) -> float:
    return h + d * u_prime

def trace_paraxial_ray(system: ThreeMirrorSystem,
                       h_init: float,
                       u_init: float) -> List[ParaxialRayState]:
    surfaces = [system.M1, system.M2, system.M3]
    separations = [system.d1, system.d2]
    n_vals = [1.0, -1.0, 1.0]  # n before each surface
    states = []
    h = h_init
    u = u_init
    for j in range(3):
        R = surfaces[j].R
        n = n_vals[j]
        c = surfaces[j].c
        u_prime, n_prime = paraxial_reflect(h, u, R, n)
        A = n * (u + h * c)
        delta_un = u_prime / n_prime - u / n
        states.append(ParaxialRayState(
            h=h, u=u, u_prime=u_prime,
            n=n, n_prime=n_prime,
            A=A, delta_un=delta_un
        ))
        if j < 2:
            h = paraxial_transfer(h, u_prime, separations[j])
            u = u_prime
    return states

def trace_system(system: ThreeMirrorSystem) -> Tuple[List[ParaxialRayState],
                                                       List[ParaxialRayState]]:
    theta = np.deg2rad(system.field_angle)
    marginal = trace_paraxial_ray(system, h_init=system.h1, u_init=0.0)
    chief = trace_paraxial_ray(system, h_init=0.0, u_init=-np.tan(theta))
    return marginal, chief

@dataclass
class SeidelCoefficients:
    S_I: float
    S_II: float
    S_III: float
    S_IV: float
    S_V: float
    S_I_per_surface: np.ndarray = field(default_factory=lambda: np.zeros(3))
    S_II_per_surface: np.ndarray = field(default_factory=lambda: np.zeros(3))
    S_III_per_surface: np.ndarray = field(default_factory=lambda: np.zeros(3))
    S_IV_per_surface: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __repr__(self):
        return (f"Seidel Coefficients:\n"
                f"  S_I   (Spherical)   = {self.S_I:+.6e}\n"
                f"  S_II  (Coma)        = {self.S_II:+.6e}\n"
                f"  S_III (Astigmatism) = {self.S_III:+.6e}\n"
                f"  S_IV  (Petzval)     = {self.S_IV:+.6e}\n"
                f"  S_V   (Distortion)  = {self.S_V:+.6e}")

def compute_seidel(system: ThreeMirrorSystem) -> SeidelCoefficients:
    marginal, chief = trace_system(system)
    surfaces = [system.M1, system.M2, system.M3]
    n_vals = [1.0, -1.0, 1.0]
    # Lagrange invariant (evaluate after M1 reflection)
    H = marginal[0].n_prime * (marginal[0].u_prime * chief[0].h -
                                chief[0].u_prime * marginal[0].h)
    S_I_total = S_II_total = S_III_total = S_IV_total = S_V_total = 0.0
    S_I_arr = np.zeros(3)
    S_II_arr = np.zeros(3)
    S_III_arr = np.zeros(3)
    S_IV_arr = np.zeros(3)
    for j in range(3):
        m = marginal[j]
        ch = chief[j]
        surf = surfaces[j]
        n_j = n_vals[j]
        c_j = surf.c
        R_j = surf.R
        K_j = surf.K
        h_j = m.h
        h_bar_j = ch.h
        A_j = m.A
        A_bar_j = n_j * (ch.u + ch.h * c_j)
        delta_un_j = m.delta_un
        # Spherical contribution (base sphere)
        Phi_j = A_j**2 * h_j * delta_un_j
        # Aspheric coefficient: a_j = 2 * n_j * K_j * c_j^3
        a_j = 2.0 * n_j * K_j * c_j**3
        # Seidel contributions
        S_I_j = Phi_j + a_j * h_j**4
        if abs(A_j) > 1e-15:
            sigma_j = A_bar_j / A_j
            S_II_j = sigma_j * Phi_j + a_j * h_j**3 * h_bar_j
            S_III_j = sigma_j**2 * Phi_j + a_j * h_j**2 * h_bar_j**2
        else:
            S_II_j = a_j * h_j**3 * h_bar_j
            S_III_j = a_j * h_j**2 * h_bar_j**2
            sigma_j = 0.0
        # Petzval
        S_IV_j = 2.0 / (n_j * R_j) if abs(R_j) > 1e-10 else 0.0
        # Distortion
        if abs(A_j) > 1e-15:
            S_V_j = sigma_j**3 * Phi_j + a_j * h_j * h_bar_j**3
            if abs(R_j) > 1e-10:
                S_V_j += sigma_j * H**2 * 2.0 / (n_j * R_j)
        else:
            S_V_j = a_j * h_j * h_bar_j**3
        S_I_total += S_I_j
        S_II_total += S_II_j
        S_III_total += S_III_j
        S_IV_total += S_IV_j
        S_V_total += S_V_j
        S_I_arr[j] = S_I_j
        S_II_arr[j] = S_II_j
        S_III_arr[j] = S_III_j
        S_IV_arr[j] = S_IV_j
    return SeidelCoefficients(
        S_I=S_I_total, S_II=S_II_total, S_III=S_III_total,
        S_IV=S_IV_total, S_V=S_V_total,
        S_I_per_surface=S_I_arr, S_II_per_surface=S_II_arr,
        S_III_per_surface=S_III_arr, S_IV_per_surface=S_IV_arr
    )

def compute_system_properties(system: ThreeMirrorSystem) -> dict:
    marginal, chief = trace_system(system)
    u3_prime = marginal[2].u_prime
    f_sys = -system.h1 / u3_prime if abs(u3_prime) > 1e-15 else float('inf')
    F_sys = abs(f_sys) / system.D1 if system.D1 != 0 else float('inf')
    h3 = marginal[2].h
    BFD = -h3 / u3_prime if abs(u3_prime) > 1e-15 else float('inf')
    m2 = (marginal[0].n * marginal[0].u_prime) / (marginal[1].n_prime * marginal[1].u_prime) \
        if abs(marginal[1].u_prime) > 1e-15 else float('inf')
    h_bar = [chief[j].h for j in range(3)]
    h_marg = [marginal[j].h for j in range(3)]
    obs_m2 = abs(marginal[1].h / marginal[0].h) if abs(marginal[0].h) > 1e-15 else 0.0
    # M3 obscuration: M3 physical extent projected at M1
    m3_phys_radius = abs(h_marg[2]) * 1.4  # 1.4x marginal height for clear aperture
    obs_m3 = m3_phys_radius / abs(h_marg[0]) if abs(h_marg[0]) > 1e-15 else 0.0
    obs_ratio = max(obs_m2, obs_m3)  # Total obscuration = whichever is larger
    seidel = compute_seidel(system)
    H = marginal[0].n_prime * (marginal[0].u_prime * chief[0].h -
                                chief[0].u_prime * marginal[0].h)
    R_petz = -H**2 / seidel.S_IV if abs(seidel.S_IV) > 1e-20 else float('inf')
    z_m2 = system.d1
    z_m3 = system.d1 + system.d2
    z_detector = z_m3 + BFD 
    return {
        'f_sys': f_sys, 'F_sys': F_sys, 'BFD': BFD, 'm2': m2,
        'h_marginal': h_marg, 'h_chief': h_bar,
        'obscuration_ratio': obs_ratio,
        'obs_m2': obs_m2, 'obs_m3': obs_m3,
        'Lagrange_invariant': H, 'R_petzval': R_petz,
        'seidel': seidel,
        'z_m2': z_m2, 'z_m3': z_m3, 'z_detector': z_detector,
    }

def print_system_summary(system: ThreeMirrorSystem):
    props = compute_system_properties(system)
    seidel = props['seidel']
    print(f"\nMirrors")
    print(f"  M1: R = {system.M1.R:+12.4f} mm   K = {system.M1.K:+8.5f}  (sphere)")
    print(f"  M2: R = {system.M2.R:+12.4f} mm   K = {system.M2.K:+8.5f}  (hyperboloid)")
    print(f"  M3: R = {system.M3.R:+12.4f} mm   K = {system.M3.K:+8.5f}  (paraboloid)")
    print(f"\nSeparations")
    print(f"  d1 (M1->M2) = {system.d1:+12.4f} mm")
    print(f"  d2 (M2->M3) = {system.d2:+12.4f} mm")
    print(f"  M3 from M1  = {system.d1 + system.d2:+12.4f} mm")
    print(f"\nSystem Parameters")
    print(f"  Primary diameter D1      = {system.D1:.2f} mm")
    print(f"  Primary focal length f1  = {system.f1:.4f} mm")
    print(f"  Primary f-number F/1     = {system.F1:.3f}")
    print(f"  System focal length      = {props['f_sys']:.4f} mm")
    print(f"  System f-number          = {props['F_sys']:.3f}")
    print(f"  Back focal distance      = {props['BFD']:.4f} mm")
    print(f"  M2 magnification         = {props['m2']:.4f}")
    print(f"  Linear obscuration ratio = {props['obscuration_ratio']:.4f}  (M2={props['obs_m2']:.4f}, M3={props['obs_m3']:.4f})")
    print(f"  Half-field angle         = {system.field_angle:.4f} deg")
    print(f"  Lagrange invariant H     = {props['Lagrange_invariant']:.6e}")
    print(f"\nMarginal Ray Heights")
    for j, h in enumerate(props['h_marginal']):
        print(f"  h{j+1} = {h:+12.6f} mm")
    print(f"\nChief Ray Heights")
    for j, h in enumerate(props['h_chief']):
        print(f"  h_bar_{j+1} = {h:+12.6f} mm")
    print(seidel)
    print(f"\nSeidel Breakdown by Surface")
    labels = ['S_I (Sph)', 'S_II (Coma)', 'S_III (Astig)', 'S_IV (Petz)']
    arrays = [seidel.S_I_per_surface, seidel.S_II_per_surface,
              seidel.S_III_per_surface, seidel.S_IV_per_surface]
    print(f"  {'':>16s}  {'M1':>14s}  {'M2':>14s}  {'M3':>14s}  {'Total':>14s}")
    for label, arr in zip(labels, arrays):
        total = arr.sum()
        print(f"  {label:>16s}  {arr[0]:+14.6e}  {arr[1]:+14.6e}  {arr[2]:+14.6e}  {total:+14.6e}")
    print(f"\nField Curvature")
    print(f"  Petzval sum  S_IV = {seidel.S_IV:+.6e}")
    print(f"  Petzval radius    = {props['R_petzval']:.4f} mm")
    petz_residual = system.M1.c - system.M2.c + system.M3.c
    print(f"  c1 - c2 + c3 = {petz_residual:+.6e}  (0 = flat field)")
    print(f"\nDetector")
    print(f"  Detector size        = {system.detector.size:.2f} mm")
    print(f"  Pixel pitch          = {system.detector.pixel_pitch*1000:.1f} um")
    print(f"  Pixels               = {system.detector.n_pixels}")
    print(f"  Detector position    = {props['z_detector']:.4f} mm from M1 vertex (z-axis)")
    bfd_val = props['BFD']
    if bfd_val < 0:
        print(f"  Focus type = Real (BFD={bfd_val:.2f}mm)")
    else:
        print(f"  Focus type = Virtual (BFD={bfd_val:.2f}mm)")
    z_det = props['z_detector']
    z_m2_pos = props['z_m2']
    z_m3_pos = props['z_m3']
    z_lo, z_hi = min(z_m2_pos, z_m3_pos), max(z_m2_pos, z_m3_pos)
    if z_lo < z_det < z_hi:
        print(f"  Detector location")
        print(f"    Clearance to M2    = {abs(z_det - z_m2_pos):.2f} mm")
        print(f"    Clearance to M3    = {abs(z_det - z_m3_pos):.2f} mm")
    else:
        print(f"  Detector location    = OUTSIDE M2-M3 gap (z={z_det:.2f}mm)")

def optimize_anastigmat(D1: float, F1: float,
                        d1_frac: float = 0.65,
                        d2_frac: float = 0.50,
                        field_angle: float = 0.5,
                        target_F_sys: Optional[float] = None,
                        flat_field: bool = False,
                        optimize_d1: bool = True,
                        detector_margin: float = 20.0,
                        max_m3_dist: float = 200.0,
                        verbose: bool = True) -> ThreeMirrorSystem:
    K1 = 0.0
    K3 = -1.0
    f1 = F1 * D1
    R1 = -2.0 * f1
    d1 = -d1_frac * f1 
    d2_init = d2_frac * f1
    def objective(x):
        if optimize_d1:
            K2, R2, R3, d2_var, d1_frac_var = x
            d1_var = -d1_frac_var * f1
        else:
            K2, R2, R3, d2_var = x
            d1_var = d1
        if R2 < 50 or R2 > abs(R1) * 4.0:
            return 1e15
        if R3 > -50 or R3 < R1 * 3.0:
            return 1e15
        if K2 > -1.01 or K2 < -1000:
            return 1e15
        if d2_var < 10 or d2_var > abs(R1) * 3.0:
            return 1e15
        z_m3_pos = d1_var + d2_var
        if abs(z_m3_pos) > max_m3_dist:
            return 1e15
        try:
            system = ThreeMirrorSystem(
                M1=MirrorSurface(R=R1, K=K1, label="M1"),
                M2=MirrorSurface(R=R2, K=K2, label="M2"),
                M3=MirrorSurface(R=R3, K=K3, label="M3"),
                d1=d1_var, d2=d2_var, D1=D1, field_angle=field_angle
            )
            seidel = compute_seidel(system)
            props = compute_system_properties(system)
            cost = seidel.S_I**2 + seidel.S_II**2 + seidel.S_III**2 + 0.1 * seidel.S_IV**2
            scale = max((D1 / 1000.0)**4, 1e-10)
            cost /= scale**2
            if flat_field:
                cost += 0.1 * seidel.S_IV**2 / scale**2
            F_target = target_F_sys if target_F_sys is not None else F1 * 2.0
            cost += 1e-4 * ((props['F_sys'] - F_target) / max(F_target, 1.0))**2
            # M3 should not be larger than M1
            h3 = abs(props['h_marginal'][2])
            if h3 > system.h1 * 1.0:
                cost += 10.0 * ((h3 / system.h1) - 1.0)**2
            # M1→M2 beam must clear M3 (M3 must fit in M1's central hole shadow)
            h1 = props['h_marginal'][0]
            h2 = props['h_marginal'][1]
            z_m3_check = d1_var + d2_var
            if abs(d1_var) > 1e-10:
                h_beam_at_m3 = abs(h1 + (h2 - h1) * z_m3_check / d1_var)
            else:
                h_beam_at_m3 = abs(h1)
            m3_physical_radius = h3 * 1.4 
            if m3_physical_radius >= h_beam_at_m3:
                return 1e15 
            # Obscuration: max of M2 shadow and M3+shroud shadow projected at M1
            obs_m2 = props['obscuration_ratio']
            obs_m3 = m3_physical_radius / abs(h1) if abs(h1) > 1e-10 else 0.0
            obs = max(obs_m2, obs_m3)
            if obs > 0.35:
                cost += 50.0 * (obs - 0.35)**2
            # Detector must be between M2 and M3: -d2 < BFD < 0
            bfd = props['BFD']
            if bfd > 0:
                return 1e15  # Virtual focus — reject
            if bfd < -(d2_var - detector_margin):
                return 1e15  # Detector at or past M2 — reject
            if bfd > -detector_margin:
                return 1e15  # Detector at or past M3 — reject

            if np.isnan(cost) or np.isinf(cost):
                return 1e15
            return cost
        except Exception:
            return 1e15
    bounds = [
        (-1000.0, -1.01),          # K2: hyperboloidal
        (50.0, abs(R1) * 4.0),     # R2: convex secondary (allow large/flat M2)
        (R1 * 3.0, -50.0),         # R3: concave, negative
        (50.0, abs(R1) * 1.5),     # d2: M2-M3 separation (also bounds max |BFD|)
    ]
    if optimize_d1:
        bounds.append((0.15, 0.85))  # d1_frac: M1-M2 separation as fraction of f1
    best_de = None
    for seed in [42, 137, 271, 577, 919]:
        result_de = differential_evolution(
            objective, bounds,
            maxiter=1500, tol=1e-14, seed=seed,
            popsize=50, mutation=(0.5, 1.5), recombination=0.9,
            polish=False
        )
        if best_de is None or result_de.fun < best_de.fun:
            best_de = result_de
    result_de = best_de
    result_nm = minimize(
        objective, result_de.x,
        method='Nelder-Mead',
        options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-20, 'adaptive': True}
    )
    result_pw = minimize(
        objective, result_nm.x,
        method='Powell',
        options={'maxiter': 20000, 'ftol': 1e-22}
    )
    if optimize_d1:
        K2_opt, R2_opt, R3_opt, d2_opt, d1_frac_opt = result_pw.x
        d1_opt = -d1_frac_opt * f1
    else:
        K2_opt, R2_opt, R3_opt, d2_opt = result_pw.x
        d1_opt = d1
    system = ThreeMirrorSystem(
        M1=MirrorSurface(R=R1, K=K1, label="M1"),
        M2=MirrorSurface(R=R2_opt, K=K2_opt, label="M2"),
        M3=MirrorSurface(R=R3_opt, K=K3, label="M3"),
        d1=d1_opt, d2=d2_opt, D1=D1, field_angle=field_angle
    )
    return system
def intersect_conic(pos: np.ndarray, direction: np.ndarray,
                    surface: MirrorSurface) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    c = surface.c
    K = surface.K
    l, m, n_dir = direction
    x0, y0, z0 = pos
    # Implicit conic: c*(x^2 + y^2) - 2*z + (1+K)*c*z^2 = 0
    A_coeff = c * (l**2 + m**2) + (1 + K) * c * n_dir**2
    B_coeff = 2 * c * (x0 * l + y0 * m) + 2 * (1 + K) * c * z0 * n_dir - 2 * n_dir
    C_coeff = c * (x0**2 + y0**2) + (1 + K) * c * z0**2 - 2 * z0
    if abs(A_coeff) < 1e-15:
        if abs(B_coeff) < 1e-15:
            return None
        t = -C_coeff / B_coeff
    else:
        discriminant = B_coeff**2 - 4 * A_coeff * C_coeff
        if discriminant < 0:
            return None
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-B_coeff + sqrt_disc) / (2 * A_coeff)
        t2 = (-B_coeff - sqrt_disc) / (2 * A_coeff)
        candidates = [t for t in [t1, t2] if t > 1e-10]  # forward propagation only
        if not candidates:
            return None
        t = min(candidates) 
    xi = x0 + l * t
    yi = y0 + m * t
    zi = z0 + n_dir * t
    dFdx = 2 * c * xi
    dFdy = 2 * c * yi
    dFdz = -2 + 2 * (1 + K) * c * zi
    normal = np.array([dFdx, dFdy, dFdz])
    norm = np.linalg.norm(normal)
    if norm < 1e-15:
        return None
    normal /= norm

    if np.dot(normal, direction) > 0:
        normal = -normal

    return np.array([xi, yi, zi]), normal

def reflect_direction(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Compute reflected ray direction."""
    d_dot_n = np.dot(direction, normal)
    reflected = direction - 2 * d_dot_n * normal
    return reflected / np.linalg.norm(reflected)

def trace_exact_ray(system: ThreeMirrorSystem,
                    pupil_x: float, pupil_y: float,
                    field_x: float = 0.0, field_y: float = 0.0,
                    _cached_BFD: Optional[float] = None
                    ) -> Optional[np.ndarray]:
    h1 = system.h1
    if _cached_BFD is not None:
        BFD = _cached_BFD
    else:
        props = compute_system_properties(system)
        BFD = props['BFD']
    x0 = pupil_x * h1
    y0 = pupil_y * h1
    r0 = np.sqrt(x0**2 + y0**2)
    z0 = system.M1.sag(np.array([r0]))[0]
    fx = np.deg2rad(field_x)
    fy = np.deg2rad(field_y)
    cos_total = np.cos(np.sqrt(fx**2 + fy**2))
    direction = np.array([np.sin(fx), np.sin(fy), cos_total])
    direction /= np.linalg.norm(direction)
    pos = np.array([x0, y0, z0])
    # Reflect off M1 — compute surface normal directly at the known hit point
    c1 = system.M1.c
    K1 = system.M1.K
    dFdx = 2.0 * c1 * pos[0]
    dFdy = 2.0 * c1 * pos[1]
    dFdz = -2.0 + 2.0 * (1.0 + K1) * c1 * pos[2]
    normal = np.array([dFdx, dFdy, dFdz])
    norm_mag = np.linalg.norm(normal)
    if norm_mag < 1e-15:
        return None
    normal /= norm_mag
    if np.dot(normal, direction) > 0:
        normal = -normal
    direction = reflect_direction(direction, normal)
    # Transfer to M2 coords
    pos_m2 = pos.copy()
    pos_m2[2] -= system.d1
    result = intersect_conic(pos_m2, direction, system.M2)
    if result is None:
        return None
    pos_m2, normal = result
    direction = reflect_direction(direction, normal)
    # Transfer to M3 coords
    pos_m3 = pos_m2.copy()
    pos_m3[2] -= system.d2
    result = intersect_conic(pos_m3, direction, system.M3)
    if result is None:
        return None
    pos_m3, normal = result
    direction = reflect_direction(direction, normal)
    # Propagate to focal plane
    if abs(direction[2]) < 1e-15:
        return None
    t_focus = (BFD - pos_m3[2]) / direction[2]
    if t_focus < 0:
        return None  # Virtual focus
    focal_pos = pos_m3 + t_focus * direction
    return focal_pos

def generate_spot_diagram(system: ThreeMirrorSystem,
                          field_angles: List[float] = None,
                          n_rings: int = 8,
                          n_arms: int = 18) -> dict:
    if field_angles is None:
        fa = system.field_angle
        field_angles = [0.0, fa * 0.33, fa * 0.67, fa]
    props = compute_system_properties(system)
    BFD = props['BFD']
    spots = {}
    for fa in field_angles:
        xs, ys = [], []
        for ring in range(1, n_rings + 1):
            r = ring / n_rings
            for arm in range(n_arms):
                theta = 2.0 * np.pi * arm / n_arms
                px = r * np.cos(theta)
                py = r * np.sin(theta)
                result = trace_exact_ray(system, px, py, field_x=0.0, field_y=fa, _cached_BFD=BFD)
                if result is not None:
                    xs.append(result[0])
                    ys.append(result[1])
        chief = trace_exact_ray(system, 0.0, 0.0, field_x=0.0, field_y=fa, _cached_BFD=BFD)
        if chief is not None and len(xs) > 0:
            xs = np.array(xs) - chief[0]
            ys = np.array(ys) - chief[1]
        else:
            xs = np.array(xs)
            ys = np.array(ys)

        spots[fa] = (xs, ys)
    return spots
def plot_system_layout(system: ThreeMirrorSystem, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    props = compute_system_properties(system)
    BFD = props['BFD']
    # Mirror z-positions in physical space (signed z relative to M1)
    z_m1 = 0.0
    z_m2 = system.d1               
    z_m3 = system.d1 + system.d2   
    z_focus = z_m3 + BFD           
    surfaces = [system.M1, system.M2, system.M3]
    z_positions = [z_m1, z_m2, z_m3]
    colors = ['#2196F3', '#FF5722', '#4CAF50']
    h_maxes = [abs(props['h_marginal'][j]) * 1.4 for j in range(3)]
    for i in range(3):
        h_maxes[i] = max(h_maxes[i], system.h1 * 0.15)
    det = system.detector
    shroud_r = det.shroud_radius
    for j, (surf, z_pos, color, h_max) in enumerate(
            zip(surfaces, z_positions, colors, h_maxes)):
        r = np.linspace(-h_max, h_max, 300)
        sag = surf.sag(np.abs(r))
        if j == 1:  
            r_cut = r.copy()
            sag_cut = sag.copy()
            mask = np.abs(r) <= shroud_r
            r_cut[mask] = np.nan
            sag_cut[mask] = np.nan
            ax.plot(z_pos + sag_cut, r_cut, color=color, linewidth=2.5,
                    label=f"M2 (R={surf.R:.0f}, cutout r={shroud_r:.0f})")
        else:
            ax.plot(z_pos + sag, r, color=color, linewidth=2.5, label=f"M{j+1} (R={surf.R:.0f})")
    for frac in np.linspace(0.3, 1.0, 5):
        h_at_m1 = frac * props['h_marginal'][0]
        h_at_m2 = frac * props['h_marginal'][1]
        h_at_m3 = frac * props['h_marginal'][2]
        z_pts = [z_m1, z_m2, z_m3, z_focus]
        h_pts = [h_at_m1, h_at_m2, h_at_m3, 0.0]
        ax.plot(z_pts, h_pts, 'gold', linewidth=0.6, alpha=0.7)
        ax.plot(z_pts, [-hp for hp in h_pts], 'gold', linewidth=0.6, alpha=0.7)
    ax.axvline(x=z_focus, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Focal plane')
    ax.scatter([z_focus], [0], color='red', s=40, zorder=5)
    shroud_len = det.shroud_length
    shroud_z_start = z_focus - shroud_len / 2.0
    shroud_rect = plt.Rectangle(
        (shroud_z_start, -shroud_r), shroud_len, 2 * shroud_r,
        fill=True, facecolor='#E0E0E0', edgecolor='purple',
        linewidth=1.5, alpha=0.4, zorder=2,
        label=f'Shroud (r={shroud_r:.0f}, L={shroud_len:.0f}mm)')
    ax.add_patch(shroud_rect)
    det_half = det.size / 2.0
    ax.plot([z_focus, z_focus], [-det_half, det_half], color='magenta',
            linewidth=3, solid_capstyle='butt', zorder=4,
            label=f'Detector ({det.size:.0f}mm)')
    ax.set_xlabel('z (mm)', fontsize=12)
    ax.set_ylabel('y (mm)', fontsize=12)
    ax.set_title('Three-Mirror Anastigmat Layout', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    return ax

def plot_seidel_bar_chart(system: ThreeMirrorSystem, ax=None):
    seidel = compute_seidel(system)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    labels = ['S_I\n(Spherical)', 'S_II\n(Coma)', 'S_III\n(Astigm.)', 'S_IV\n(Petzval)']
    x = np.arange(len(labels))
    width = 0.2
    m1_vals = [seidel.S_I_per_surface[0], seidel.S_II_per_surface[0],
               seidel.S_III_per_surface[0], seidel.S_IV_per_surface[0]]
    m2_vals = [seidel.S_I_per_surface[1], seidel.S_II_per_surface[1],
               seidel.S_III_per_surface[1], seidel.S_IV_per_surface[1]]
    m3_vals = [seidel.S_I_per_surface[2], seidel.S_II_per_surface[2],
               seidel.S_III_per_surface[2], seidel.S_IV_per_surface[2]]
    totals = [seidel.S_I, seidel.S_II, seidel.S_III, seidel.S_IV]
    ax.bar(x - 1.5*width, m1_vals, width, label='M1 (sphere)', color='#2196F3', alpha=0.8)
    ax.bar(x - 0.5*width, m2_vals, width, label='M2 (hyperboloid)', color='#FF5722', alpha=0.8)
    ax.bar(x + 0.5*width, m3_vals, width, label='M3 (paraboloid)', color='#4CAF50', alpha=0.8)
    ax.bar(x + 1.5*width, totals, width, label='Total', color='#333333', alpha=0.9)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Seidel Coefficient', fontsize=11)
    ax.set_title('Seidel Aberration Breakdown by Surface', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    return ax

def generate_full_report(system: ThreeMirrorSystem, save_path: str = None):
    props = compute_system_properties(system)
    seidel = props['seidel']
    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    plot_system_layout(system, ax=ax1)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_seidel_bar_chart(system, ax=ax2)
    ax3 = fig.add_subplot(gs[1, 1])
    surfaces = [system.M1, system.M2, system.M3]
    colors = ['#2196F3', '#FF5722', '#4CAF50']
    klabels = ['M1 (K=0)', f'M2 (K={system.M2.K:.3f})', 'M3 (K=-1)']
    for j, (surf, color, kl) in enumerate(zip(surfaces, colors, klabels)):
        h_max = abs(props['h_marginal'][j])
        if h_max < 1e-6:
            h_max = system.h1 * 0.3
        r = np.linspace(0, h_max, 200)
        sag = surf.sag(r)
        ax3.plot(r, sag, color=color, linewidth=2, label=kl)
    ax3.set_xlabel('Radial height r (mm)', fontsize=11)
    ax3.set_ylabel('Sag (mm)', fontsize=11)
    ax3.set_title('Mirror Sag Profiles', fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax4 = fig.add_subplot(gs[2, 0])
    for j, (surf, color) in enumerate(zip(surfaces, colors)):
        h_max = abs(props['h_marginal'][j])
        if h_max < 1e-6:
            h_max = system.h1 * 0.3
        r = np.linspace(1e-6, h_max, 200)
        c = surf.c
        K = surf.K
        if abs(c) > 1e-15 and abs(K) > 1e-10:
            sag_conic = surf.sag(r)
            arg_sph = np.maximum(1 - c**2 * r**2, 0)
            sag_sphere = c * r**2 / (1 + np.sqrt(arg_sph))
            departure = (sag_conic - sag_sphere) * 1e3  # microns
            ax4.plot(r, departure, color=color, linewidth=2, label=f"M{j+1}")

    ax4.set_xlabel('Radial height r (mm)', fontsize=11)
    ax4.set_ylabel('Aspherical departure (um)', fontsize=11)
    ax4.set_title('Departure from Best-Fit Sphere', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax5 = fig.add_subplot(gs[2, 1])
    field_angles_sweep = np.linspace(0, system.field_angle, 30)
    sa_vals, coma_vals, astig_vals = [], [], []
    for fa in field_angles_sweep:
        sys_temp = ThreeMirrorSystem(
            M1=system.M1, M2=system.M2, M3=system.M3,
            d1=system.d1, d2=system.d2, D1=system.D1, field_angle=fa
        )
        s = compute_seidel(sys_temp)
        sa_vals.append(abs(s.S_I))
        coma_vals.append(abs(s.S_II))
        astig_vals.append(abs(s.S_III))

    ax5.semilogy(field_angles_sweep, sa_vals, 'b-', linewidth=2, label='|S_I| (Spherical)')
    ax5.semilogy(field_angles_sweep, coma_vals, 'r-', linewidth=2, label='|S_II| (Coma)')
    ax5.semilogy(field_angles_sweep, astig_vals, 'g-', linewidth=2, label='|S_III| (Astigmatism)')
    ax5.set_xlabel('Half-field angle (deg)', fontsize=11)
    ax5.set_ylabel('|Seidel coefficient|', fontsize=11)
    ax5.set_title('Aberration vs Field Angle', fontsize=13)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    fa = system.field_angle
    spot_fields = [0.0, fa * 0.5, fa]
    spots = generate_spot_diagram(system, field_angles=spot_fields, n_rings=10, n_arms=24)
    wavelength = 0.55e-3  # mm
    airy_radius = 1.22 * wavelength * props['F_sys']
    pixel_size_um = system.detector.pixel_pitch * 1000  # microns
    for i, (fa_val, (xs, ys)) in enumerate(spots.items()):
        if i >= 2:
            break
        ax = fig.add_subplot(gs[3, i])
        if len(xs) > 0:
            ax.scatter(xs * 1000, ys * 1000, s=2, c='navy', alpha=0.5)
            theta_circ = np.linspace(0, 2 * np.pi, 100)
            ax.plot(airy_radius * 1000 * np.cos(theta_circ),
                    airy_radius * 1000 * np.sin(theta_circ),
                    'r--', linewidth=1.5, label=f'Airy ({airy_radius*1e6:.1f}um)')
            ax.add_patch(plt.Rectangle(
                (-pixel_size_um/2, -pixel_size_um/2), pixel_size_um, pixel_size_um,
                fill=False, edgecolor='green', linestyle='--', linewidth=1.0,
                label=f'Pixel ({pixel_size_um:.0f}um)'))
            rms = np.sqrt(np.mean(xs**2 + ys**2)) * 1000
            ax.set_title(f'Spot @ {fa_val:.3f} deg (RMS={rms:.2f}um)', fontsize=11)
        else:
            ax.set_title(f'Spot @ {fa_val:.3f} deg (no rays)', fontsize=11)
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Three-Mirror Anastigmat Design Report\n'
                 f'Sphere / Hyperboloid / Paraboloid  |  '
                 f'D1={system.D1:.0f}mm  F/{props["F_sys"]:.2f}  '
                 f'FOV=+/-{system.field_angle:.2f} deg',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Report saved to {save_path}")
    return fig
def main():
    print("THREE-MIRROR ANASTIGMAT")
    print("Mirror shapes: Sphere -> Hyperboloid -> Paraboloid")
    print("\nSystem Requirements")
    D1 = 1500.0
    F1 = 3.0
    field_angle = 0.3
    print(f"  Primary diameter: {D1:.0f} mm")
    print(f"  Primary f-number: F/{F1:.1f}")
    print(f"  Half-field angle: {field_angle:.2f} deg\n")
    print("\nAnastigmatic Optimization Running...")
    system = optimize_anastigmat(
        D1=D1, F1=F1, d1_frac=0.65, d2_frac=0.50,
        field_angle=field_angle, flat_field=False, verbose=True,
        max_m3_dist=1000.0
    )
    print("\nDesign Summary")
    print_system_summary(system)
    generate_full_report(system, save_path='c:/Users/smsin/Downloads/tma_report.png')
    props = compute_system_properties(system)
    return system, props
if __name__ == '__main__':
    system, props = main()