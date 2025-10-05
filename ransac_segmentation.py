import torch
import random


# Define the function to fit a plane using RANSAC
def fit_plane_ransac(
    points, max_iterations=1000, distance_threshold=0.01, min_inliers=3
):
    """
    Fit a plane to a set of 3D points using the RANSAC algorithm with GPU acceleration.

    Args:
    - points: Nx3 list or numpy array of 3D points.
    - max_iterations: Maximum number of iterations for RANSAC.
    - distance_threshold: Distance threshold to consider a point as an inlier.
    - min_inliers: Minimum number of inliers to consider a model valid.

    Returns:
    - best_plane: List of plane coefficients (a, b, c, d).
    - best_inliers: List of indices of inliers.
    - best_outliers: List of indices of outliers.
    """
    # Convert points to a PyTorch tensor and move to GPU if available
    points = torch.tensor(points, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = points.to(device)

    def compute_plane(p1, p2, p3):
        """Compute the plane equation ax + by + cz + d = 0 from three points."""
        v1 = p2 - p1
        v2 = p3 - p1
        normal = torch.linalg.cross(v1, v2)
        a, b, c = normal
        d = -torch.dot(normal, p1)
        norm = torch.norm(normal)
        return a / norm, b / norm, c / norm, d / norm

    def compute_distances_to_plane(points, plane):
        """Compute distances of all points to a plane in a vectorized manner."""
        a, b, c, d = plane
        distances = torch.abs(
            points @ torch.tensor([a, b, c], device=points.device) + d
        )
        distances /= torch.sqrt(a**2 + b**2 + c**2)
        return distances

    best_plane = None
    best_inliers = []
    best_outliers = []
    num_points = points.shape[0]

    for _ in range(max_iterations):
        # Randomly sample three points for RANSAC
        sample_indices = random.sample(range(num_points), 3)
        p1, p2, p3 = points[sample_indices]

        # Compute the plane using the selected points
        plane = compute_plane(p1, p2, p3)

        # Compute distances for all points and find inliers
        distances = compute_distances_to_plane(points, plane)
        inliers = torch.nonzero(distances < distance_threshold, as_tuple=True)[0]

        if len(inliers) > min_inliers and len(inliers) > len(best_inliers):
            # Refine the plane using inliers
            inlier_points = points[inliers]
            centroid = torch.mean(inlier_points, dim=0)
            centered_points = inlier_points - centroid

            # Use SVD for plane refinement
            _, _, V = torch.linalg.svd(centered_points, full_matrices=False)
            normal = V[-1]
            d = -torch.dot(centroid, normal)
            refined_plane = (normal[0], normal[1], normal[2], d)

            # Recompute inliers and outliers for the refined plane
            distances = compute_distances_to_plane(points, refined_plane)
            refined_inliers = torch.nonzero(
                distances < distance_threshold, as_tuple=True
            )[0]
            refined_outliers = torch.nonzero(
                distances >= distance_threshold, as_tuple=True
            )[0]

            if len(refined_inliers) > len(best_inliers):
                best_plane = refined_plane
                best_inliers = refined_inliers
                best_outliers = refined_outliers

    # Move results back to CPU for output
    best_plane = [x.cpu().item() for x in best_plane]
    best_inliers = best_inliers.cpu().tolist()
    best_outliers = best_outliers.cpu().tolist()
    torch.cuda.empty_cache()

    return best_plane, best_inliers, best_outliers

# Example usage
if __name__ == "__main__":
    # Generate some synthetic data
    import numpy as np

    # Create a plane z = 0.5x + 0.2y + 1 with some noise
    num_points = 100
    X = np.random.uniform(-10, 10, num_points)
    Y = np.random.uniform(-10, 10, num_points)
    Z = 0.5 * X + 0.2 * Y + 1 + np.random.normal(0, 0.5, num_points)
    points = np.vstack((X, Y, Z)).T

    # Add some outliers
    num_outliers = 20
    outliers = np.random.uniform(-10, 10, (num_outliers, 3))
    points = np.vstack((points, outliers))

    # Fit the plane using RANSAC
    best_plane, best_inliers, best_outliers = fit_plane_ransac(
        points, max_iterations=1000, distance_threshold=0.5, min_inliers=30
    )

    print("Best Plane Coefficients (a, b, c, d):", best_plane)
    print("Number of Inliers:", len(best_inliers))
    print("Number of Outliers:", len(best_outliers))