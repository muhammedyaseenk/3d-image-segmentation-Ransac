import torch
import random

# Define the function to fit a plane using batched RANSAC
def fit_plane_ransac_batch(
    points, max_iterations=1000, distance_threshold=0.01, min_inliers=3, iterations_per_batch=40, epsilon=1e-8
):
    """
    Find the best equation for a plane using a batched RANSAC approach.

    This function fits a plane to a 3D point cloud using a RANSAC-like method,
    processing multiple iterations in parallel for efficiency.

    Args:
    - points: Nx3 list or numpy array of 3D points.
    - max_iterations: Maximum number of iterations for the RANSAC algorithm.
    - distance_threshold: Distance threshold to consider a point as an inlier.
    - min_inliers: Minimum number of inliers to consider a model valid.
    - iterations_per_batch: Number of iterations to process in parallel per batch.
    - epsilon: Small value to avoid division by zero.

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

    for start_idx in range(0, max_iterations, iterations_per_batch):
        end_idx = min(start_idx + iterations_per_batch, max_iterations)
        current_batch_size = end_idx - start_idx

        # Sample 3 random points for each iteration in the batch
        rand_pt_idx = torch.randint(0, num_points, (current_batch_size, 3), device=device)
        sampled_points = points[rand_pt_idx]  # (batch_size, 3, 3)

        # Compute vectors vecA and vecB
        vec_A = sampled_points[:, 1, :] - sampled_points[:, 0, :]  # (batch_size, 3)
        vec_B = sampled_points[:, 2, :] - sampled_points[:, 0, :]  # (batch_size, 3)

        # Compute cross product vecC, which is the normal to the plane
        vec_C = torch.cross(vec_A, vec_B, dim=1)  # (batch_size, 3)

        # Normalize vecC to get the unit normal vector
        vec_C = vec_C / (torch.norm(vec_C, dim=1, keepdim=True) + epsilon)  # (batch_size, 3)

        # Compute the constant term k for each plane
        k = -torch.einsum("ij,ij->i", vec_C, sampled_points[:, 1, :])  # (batch_size,)

        # Plane equation coefficients (Ax + By + Cz + D)
        plane_eq = torch.cat([vec_C, k.unsqueeze(1)], dim=1)  # (batch_size, 4)

        # Compute distances of all points to each plane in the batch
        dist_pts = (
            plane_eq[:, 0:1] * points[:, 0].unsqueeze(0)
            + plane_eq[:, 1:2] * points[:, 1].unsqueeze(0)
            + plane_eq[:, 2:3] * points[:, 2].unsqueeze(0)
            + plane_eq[:, 3:4]
        ) / torch.sqrt(
            plane_eq[:, 0] ** 2 + plane_eq[:, 1] ** 2 + plane_eq[:, 2] ** 2
        ).unsqueeze(1)

        # Inlier mask: points where distance <= threshold
        inlier_mask = torch.abs(dist_pts) <= distance_threshold  # (batch_size, num_pts)
        inlier_counts = inlier_mask.sum(dim=1)  # (batch_size,)

        # Find the best iteration in this batch
        best_in_batch_idx = torch.argmax(inlier_counts)  # Best plane in this batch
        best_inlier_count_in_batch = inlier_counts[best_in_batch_idx].item()

        if best_inlier_count_in_batch >= min_inliers and best_inlier_count_in_batch > len(best_inliers):
            best_inlier_indices = torch.where(inlier_mask[best_in_batch_idx])[0]
            best_eq = plane_eq[best_in_batch_idx]

            # Refine the plane using inliers
            inlier_points = points[best_inlier_indices]
            centroid = torch.mean(inlier_points, dim=0)
            centered_points = inlier_points - centroid

            # Use SVD for plane refinement
            _, _, V = torch.linalg.svd(centered_points, full_matrices=False)
            normal = V[-1]
            d = -torch.dot(centroid, normal)
            refined_plane = (normal[0], normal[1], normal[2], d)

            # Recompute inliers and outliers for the refined plane
            distances = compute_distances_to_plane(points, refined_plane)
            refined_inliers = torch.nonzero(distances < distance_threshold, as_tuple=True)[0]
            refined_outliers = torch.nonzero(distances >= distance_threshold, as_tuple=True)[0]

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



# Example usage:
if __name__ == "__main__":
    # Generate some random 3D points for testing
    import numpy as np

    # Create a plane: 0.5x + 0.3y + 0.2z + 1 = 0
    num_inliers = 100
    num_outliers = 20
    inlier_points = np.random.rand(num_inliers, 2)
    z_inliers = (-1 - 0.5 * inlier_points[:, 0] - 0.3 * inlier_points[:, 1]) / 0.2
    inlier_points = np.hstack((inlier_points, z_inliers.reshape(-1, 1)))

    outlier_points = np.random.rand(num_outliers, 3) * 10  # Random outliers far from the plane

    all_points = np.vstack((inlier_points, outlier_points))

    best_plane, best_inliers, best_outliers = fit_plane_ransac_batch(
        all_points, max_iterations=1000, distance_threshold=0.05, min_inliers=50, iterations_per_batch=50
    )

    print("Best Plane Coefficients (a, b, c, d):", best_plane)
    print("Number of Inliers:", len(best_inliers))
    print("Number of Outliers:", len(best_outliers))