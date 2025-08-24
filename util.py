import numpy as np

def catmull_rom_spline(P0, P1, P2, P3, n_points=20):
    t = np.linspace(0, 1, n_points)[:, None]
    a = 2*P1
    b = P2 - P0
    c = 2*P0 - 5*P1 + 4*P2 - P3
    d = -P0 + 3*P1 - 3*P2 + P3
    return 0.5 * (a + b*t + c*t**2 + d*t**3)

def smooth_closed_loop(points, n_points_per_segment=20):
    smoothed = []
    N = len(points)
    for i in range(N):
        P0 = points[(i-1) % N]
        P1 = points[i]
        P2 = points[(i+1) % N]
        P3 = points[(i+2) % N]
        smoothed.append(catmull_rom_spline(P0, P1, P2, P3, n_points_per_segment))
    all_pts = np.vstack(smoothed)
    if all_pts.size == 0:
        smoothed = [all_pts]
    else:
        keep = [0]
        for i in range(1, len(all_pts)):
            if np.linalg.norm(all_pts[i] - all_pts[keep[-1]]) >= 0.1:
                keep.append(i)
        filtered = all_pts[keep]
        if len(filtered) > 1 and np.linalg.norm(filtered[0] - filtered[-1]) < 0.1:
            filtered = filtered[:-1]
        smoothed = [filtered]
    return np.vstack(smoothed)
