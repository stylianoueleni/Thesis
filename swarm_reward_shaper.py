import numpy as np

class SwarmRewardShaper:
    """
    Parameters
    ----------
    num_uavs          : int   - swarm size
    desired_spacing   : float - target centre-to-centre separation  [m]
    cohesion_weight   : float - penalty weight for centroid deviation
    separation_weight : float - weight for spacing Gaussian well
    alignment_weight  : float - weight for heading-agreement bonus
    separation_min    : float - hard danger-zone threshold  [m]
                                (set to 2*body_radius + uav_separation)
    """

    def __init__(
        self,
        num_uavs:          int,
        desired_spacing:   float = 6.0,
        cohesion_weight:   float = 1.5,
        separation_weight: float = 4.0,
        alignment_weight:  float = 0.8,
        separation_min:    float = 3.0,
    ):
        self.num_uavs          = num_uavs
        self.desired_spacing   = float(desired_spacing)
        self.cohesion_weight   = float(cohesion_weight)
        self.separation_weight = float(separation_weight)
        self.alignment_weight  = float(alignment_weight)
        self.separation_min    = float(separation_min)

    def shape(
        self,
        positions: np.ndarray,
        headings: np.ndarray,
        dones: np.ndarray) -> np.ndarray:
        N = self.num_uavs
        shaped = np.zeros(N, dtype=np.float64)

        active_idx = np.where(~dones)[0]
        if len(active_idx) < 2:
            return shaped
        
        centroid = positions[active_idx].mean(axis=0)

        for i in active_idx:
            cohesion_r = 0.0
            separation_r = 0.0
            alignment_r = 0.0
            n_neighbors = 0

            dist_centroid = np.linalg.norm(positions[i] - centroid)
            cohesion_r = -self.cohesion_weight * dist_centroid

            for j in active_idx:
                if j ==i:
                    continue

                d_ij = np.linalg.norm(positions[i] - positions[j])

                delta = d_ij - self.desired_spacing
                separation_r += (-self.separation_weight * (delta ** 2) / (self.desired_spacing ** 2))

                if d_ij < self.separation_min:
                    separation_r -= self.separation_weight * 10.0

                alignment_r += np.cos(headings[i] - headings[j])
                n_neighbors += 1

            if n_neighbors > 0:
                alignment_r = self.alignment_weight * (alignment_r / n_neighbors)

            shaped[i] = cohesion_r + separation_r + alignment_r

        return shaped