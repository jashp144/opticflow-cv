"""Motion vector spatial-velocity clustering."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from opticflow_cv.core.models import MotionVector


@dataclass
class MotionCluster:
    """Represents a coherent cluster of motion vectors sharing spatial proximity and velocity."""

    cluster_id: int
    centroid: tuple[float, float]
    average_velocity: tuple[float, float]
    average_speed: float
    member_count: int
    bounding_box: tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    vectors: list[MotionVector] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize cluster metadata."""
        return {
            "cluster_id": self.cluster_id,
            "centroid": self.centroid,
            "average_velocity": self.average_velocity,
            "average_speed": self.average_speed,
            "member_count": self.member_count,
            "bounding_box": self.bounding_box,
        }


class MotionClusterer:
    """Clusters motion vectors into distinct moving objects based on spatial and velocity affinity."""

    def __init__(
        self,
        eps: float = 30.0,
        min_samples: int = 3,
        velocity_similarity: float = 0.5,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")

        self.eps = eps
        self.min_samples = min_samples
        self.velocity_similarity = velocity_similarity

    def cluster(self, vectors: list[MotionVector]) -> list[MotionCluster]:
        """Group motion vectors into spatial-velocity clusters using density connectivity."""
        if not vectors:
            return []

        n = len(vectors)
        visited = [False] * n
        clusters: list[MotionCluster] = []
        cluster_id = 1

        for i in range(n):
            if visited[i]:
                continue

            # Find density-reachable neighbors
            neighbors = self._find_neighbors(i, vectors)
            if len(neighbors) < self.min_samples:
                continue

            # Expand cluster
            cluster_indices = set(neighbors)
            queue = list(neighbors)
            visited[i] = True

            while queue:
                curr_idx = queue.pop(0)
                if not visited[curr_idx]:
                    visited[curr_idx] = True
                    curr_neighbors = self._find_neighbors(curr_idx, vectors)
                    if len(curr_neighbors) >= self.min_samples:
                        for n_idx in curr_neighbors:
                            if n_idx not in cluster_indices:
                                cluster_indices.add(n_idx)
                                queue.append(n_idx)

            # Build MotionCluster from indices
            cluster_vecs = [vectors[idx] for idx in cluster_indices]
            cluster_obj = self._build_cluster(cluster_id, cluster_vecs)
            clusters.append(cluster_obj)
            cluster_id += 1

        return clusters

    def _find_neighbors(self, index: int, vectors: list[MotionVector]) -> list[int]:
        target = vectors[index]
        neighbors: list[int] = []

        for j, other in enumerate(vectors):
            # Spatial distance
            dist = math.hypot(target.x - other.x, target.y - other.y)
            if dist > self.eps:
                continue

            # Velocity vector cosine similarity / direction consistency
            mag_prod = target.magnitude * other.magnitude
            if mag_prod > 1e-6:
                cos_sim = (target.vx * other.vx + target.vy * other.vy) / mag_prod
                if cos_sim < self.velocity_similarity:
                    continue

            neighbors.append(j)

        return neighbors

    def _build_cluster(self, cluster_id: int, cluster_vecs: list[MotionVector]) -> MotionCluster:
        count = len(cluster_vecs)
        mean_x = sum(v.x for v in cluster_vecs) / count
        mean_y = sum(v.y for v in cluster_vecs) / count
        mean_vx = sum(v.vx for v in cluster_vecs) / count
        mean_vy = sum(v.vy for v in cluster_vecs) / count
        mean_speed = sum(v.magnitude for v in cluster_vecs) / count

        min_x = int(min(v.x for v in cluster_vecs))
        min_y = int(min(v.y for v in cluster_vecs))
        max_x = int(max(v.x for v in cluster_vecs))
        max_y = int(max(v.y for v in cluster_vecs))

        return MotionCluster(
            cluster_id=cluster_id,
            centroid=(mean_x, mean_y),
            average_velocity=(mean_vx, mean_vy),
            average_speed=mean_speed,
            member_count=count,
            bounding_box=(min_x, min_y, max_x, max_y),
            vectors=cluster_vecs,
        )
