import base64
import collections
import heapq
import json
import math
import os
import random
import traceback
from io import BytesIO
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import recast
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

try:
    from ..src.general_utils import rank0_log
except ImportError:
    from src.general_utils import rank0_log


# =================================================================================
# Part 1: NavMesh Graph & Exploration Logic
# =================================================================================

class NavMeshGraph:
    def __init__(self, verts, faces, sample_spacing=0.25):
        self.verts = np.array(verts)
        self.faces = np.array(faces)
        self.spacing = sample_spacing


        self.mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces, process=False)
        # self.nav_scale = float(np.linalg.norm(self.mesh.bounds[1] - self.mesh.bounds[0]))

        self.centers, self.adjacency = self._build_dense_graph_trimesh(self.spacing)

        print(f"[NavMesh] Built dense graph: {len(self.centers)} nodes.")


        self.tree = cKDTree(self.centers)

    def _build_dense_graph_trimesh(self, spacing):
        """
        Sample the surface, erode boundary points, and build the graph.
        """

        total_area = self.mesh.area
        area_per_point = spacing ** 2
        oversample_factor = 1.5
        n_samples = int((total_area / area_per_point) * oversample_factor)
        n_samples = max(n_samples, len(self.faces))

        print(f"[NavMesh] Mesh Area: {total_area:.2f}, Target Samples: {n_samples}")



        try:
            samples, _ = trimesh.sample.sample_surface(self.mesh, n_samples)
        except Exception as e:
            print(f"[NavMesh] Error in trimesh sampling: {e}. Falling back to vertices.")
            samples = self.verts


        erosion_dist = spacing
        samples = self._erode_points_by_mesh_boundary(samples, spacing, min_distance=erosion_dist)




        if len(samples) > 600_000:
            samples = samples[np.random.choice(len(samples), size=600_000, replace=False)]
        dense_points = samples


        _, unique_indices = np.unique(np.round(dense_points, 4), axis=0, return_index=True)
        dense_points = dense_points[unique_indices]



        tree = cKDTree(dense_points)

        search_radius = spacing * 5.0
        pairs = np.array(list(tree.query_pairs(r=search_radius)))

        if len(pairs) > 0:

            p1 = dense_points[pairs[:, 0]]  # (N_pairs, 3)
            p2 = dense_points[pairs[:, 1]]  # (N_pairs, 3)


            diffs = p1 - p2
            dists = np.linalg.norm(diffs, axis=1)


            height_mask = np.abs(diffs[:, 1]) <= 0.5


            valid_pairs = pairs[height_mask]
            valid_dists = dists[height_mask]


            adjacency = collections.defaultdict(list)
            for idx in range(len(valid_pairs)):
                i, j = valid_pairs[idx]
                d = valid_dists[idx]
                adjacency[i].append((j, d))
                adjacency[j].append((i, d))
        else:
            adjacency = collections.defaultdict(list)

        return dense_points, adjacency

    def _erode_points_by_mesh_boundary(self, points, spacing, min_distance):
        """
        Optimization: avoid Python geometry loops by discretizing the boundary into a point cloud and using a KDTree for fast lookup.
        """


        boundary_groups = trimesh.grouping.group_rows(self.mesh.edges_sorted, require_count=1)
        boundary_edges = self.mesh.edges[boundary_groups]

        if len(boundary_edges) == 0:
            # print("[NavMesh] Closed mesh (no boundary), skipping erosion.")
            return points

        boundary_vertices = self.mesh.vertices[boundary_edges]  # (M, 2, 3)

        boundary_samples = []
        edge_vectors = boundary_vertices[:, 1] - boundary_vertices[:, 0]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)


        sample_step = min_distance * 0.5
        if sample_step < 1e-4: sample_step = 1e-4

        counts = (edge_lengths / sample_step).astype(int) + 1

        for i, count in enumerate(counts):

            if count < 2: count = 2
            t = np.linspace(0, 1, count)
            edge_points = boundary_vertices[i, 0] + np.outer(t, edge_vectors[i])
            boundary_samples.append(edge_points)

        if not boundary_samples:
            return points

        boundary_cloud = np.vstack(boundary_samples)


        boundary_tree = cKDTree(boundary_cloud)


        dists, _ = boundary_tree.query(points, k=1, workers=-1)


        mask = dists >= min_distance

        removed_count = len(points) - mask.sum()
        if removed_count > 0:
            print(f"[NavMesh] Boundary Erosion: Removed {removed_count} points within {min_distance:.3f}m of edges.")

        return points[mask]

    def find_nearest_node(self, pos):
        if len(self.centers) == 0: return -1, None
        xz_dists = np.linalg.norm(self.centers[:, [0, 2]] - pos[[0, 2]], axis=1)
        idx = int(np.argmin(xz_dists))
        return idx, self.centers[idx]

    def compute_dijkstra(self, start_node_idx):
        """
        Compute Dijkstra distances over the full graph.
        """
        num_nodes = len(self.centers)
        distances = {node: float('inf') for node in range(num_nodes)}
        distances[start_node_idx] = 0
        predecessors = {start_node_idx: None}
        pq = [(0, start_node_idx)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > distances[u]: continue

            for v, weight in self.adjacency[u]:
                new_dist = d + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(pq, (new_dist, v))
        return distances, predecessors

    def get_paths_to_reconstruct(self, nav_graph, start_face_idx, destination_pairs_yup, distances, predecessors):
        """
        Generate two-stage reconstruction paths. If the first segment (Start -> Near) has an unsuitable angle, skip it and try Start -> Far directly.
        """
        reconstruct_paths = []
        start_origin = np.array([0.0, 0.0, 0.0])


        reachable_indices_start = [i for i in range(len(nav_graph.centers)) if distances[i] != float('inf')]
        if not reachable_indices_start:
            return [(None, None)] * len(destination_pairs_yup)

        reachable_centers_start = nav_graph.centers[reachable_indices_start]
        reachable_map_start = np.array(reachable_indices_start)


        def _generate_segment(current_start_pt, target_pt, other_pt, current_distances, current_predecessors, reachable_centers, reachable_map):
            """
            Args:
                target_pt: Target point for this path segment.
                other_pt: Reference point used to determine direction.
            """


            vec_ref = other_pt[[0, 2]] - target_pt[[0, 2]]
            norm_ref = np.linalg.norm(vec_ref)



            vec_faces = target_pt[[0, 2]] - reachable_centers[:, [0, 2]]
            norm_faces = np.linalg.norm(vec_faces, axis=1)


            valid_norms = (norm_ref > 1e-6) & (norm_faces > 1e-6)
            if not np.any(valid_norms):
                return None, None

            cos_angles = np.zeros(len(reachable_centers))
            dot_products = np.dot(vec_faces[valid_norms], vec_ref)
            cos_angles[valid_norms] = dot_products / (norm_faces[valid_norms] * norm_ref)
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(cos_angles))


            angle_mask = (angles_deg < 90.0) & valid_norms
            candidate_indices = np.where(angle_mask)[0]

            if len(candidate_indices) == 0:
                return None, None





            candidates_xz = reachable_centers[candidate_indices][:, [0, 2]]
            diff_candidates = candidates_xz - target_pt[[0, 2]]

            dist_sq_candidates = np.sum(diff_candidates ** 2, axis=1)
            best_local = np.argmin(dist_sq_candidates)

            best_global = candidate_indices[best_local]
            end_node = reachable_map[best_global]


            raw_path_centers = []
            curr = end_node
            safety_count = 0
            max_steps = len(current_predecessors)

            while curr is not None and safety_count < max_steps:
                raw_path_centers.append(nav_graph.centers[curr])
                if isinstance(current_predecessors, dict):
                    curr = current_predecessors.get(curr)
                else:
                    pred = current_predecessors[curr]
                    curr = pred if pred != -9999 else None
                safety_count += 1

            raw_path_centers.reverse()


            last_nav_center = nav_graph.centers[end_node]
            mid_point = last_nav_center * 0.9 + target_pt * 0.1
            raw_path_centers.append(mid_point)


            if hasattr(self, 'fix_start_direction'):
                optimized_path = self.fix_start_direction(current_start_pt, raw_path_centers)
            else:
                optimized_path = [current_start_pt] + raw_path_centers

            return np.array(optimized_path), end_node


        for (p_near, p_far) in destination_pairs_yup:

            # -------------------------------------------------

            # -------------------------------------------------
            path1, node_near = _generate_segment(
                start_origin,
                p_near,  # Target
                p_far,  # Other (Ref)
                distances,
                predecessors,
                reachable_centers_start,
                reachable_map_start
            )

            path2 = None

            # -------------------------------------------------

            # -------------------------------------------------
            if path1 is not None:




                dists_2, preds_2 = nav_graph.compute_dijkstra(node_near)
                reachable_indices_2 = [i for i in range(len(nav_graph.centers)) if dists_2[i] != float('inf')]

                if reachable_indices_2:
                    reachable_centers_2 = nav_graph.centers[reachable_indices_2]
                    reachable_map_2 = np.array(reachable_indices_2)

                    path2, _ = _generate_segment(
                        path1[-1],  # Start from end of path1
                        p_far,  # Target
                        p_near,  # Other (Ref)
                        dists_2,
                        preds_2,
                        reachable_centers_2,
                        reachable_map_2
                    )

            else:




                path2, _ = _generate_segment(
                    start_origin,  # Start from original start
                    p_far,  # Target
                    p_near,
                    distances,  # Original distances
                    predecessors,  # Original predecessors
                    reachable_centers_start,  # Original reachable map
                    reachable_map_start
                )




            reconstruct_paths.append((path1, path2))

        return reconstruct_paths

    def get_paths_to_targets(self, nav_graph, start_face_idx, target_points_yup, distances, predecessors, target_scales=None):
        target_paths = []
        actual_radii = []
        start_pt = np.array([0.0, 0.0, 0.0])

        reachable_indices = [i for i in range(len(nav_graph.centers)) if distances[i] != float('inf')]
        if not reachable_indices:
            return [None] * len(target_points_yup), [0.0] * len(target_points_yup)

        reachable_centers = nav_graph.centers[reachable_indices]
        reachable_map = np.array(reachable_indices)

        for i, target_pt in enumerate(target_points_yup):

            dist_start_to_target = np.linalg.norm(target_pt[[0, 2]] - start_pt[[0, 2]])
            obj_scale = target_scales[i] if (target_scales is not None and i < len(target_scales)) else 0.5
            obs_radius = min(obj_scale, dist_start_to_target)
            actual_radii.append(obs_radius)


            vec_st = (target_pt - start_pt);
            vec_st[1] = 0
            angle_st = math.atan2(vec_st[2], vec_st[0])
            ideal_angles = [angle_st + math.pi / 2, angle_st - math.pi / 2]

            vec_t_start = (start_pt - target_pt);
            vec_t_start[1] = 0


            def ang_diff(a1, a2):
                return abs(math.atan2(math.sin(a1 - a2), math.cos(a1 - a2)))

            best_node_idx = -1
            best_score = -float('inf')
            final_safe_pt = None


            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(nav_graph.mesh)
            search_angles = np.linspace(0, 2 * math.pi, 72)


            for ang in search_angles:
                query_pos = target_pt + np.array([obs_radius * math.cos(ang), 0, obs_radius * math.sin(ang)])
                query_pos[1] = target_pt[1]


                if intersector is not None:
                    ray_origin_down = np.array([[query_pos[0], 500.0, query_pos[2]]])
                    ray_dir_down = np.array([[0.0, -1.0, 0.0]])
                    if not intersector.intersects_any(ray_origins=ray_origin_down, ray_directions=ray_dir_down)[0]:
                        continue


                dists = np.linalg.norm(reachable_centers[:, [0, 2]] - query_pos[[0, 2]], axis=1)
                idx_local = np.argmin(dists)
                nearest_nav_node_pt = reachable_centers[idx_local]


                vec_t_sample = (query_pos - target_pt);
                vec_t_sample[1] = 0
                if np.dot(vec_t_start, vec_t_sample) < -1e-4:
                    continue


                diff = min(ang_diff(ang, ideal_angles[0]), ang_diff(ang, ideal_angles[1]))

                # ========================================================

                # ========================================================
                los_ratio = 1.0
                if intersector is not None:
                    num_samples = 50

                    samples = np.linspace(query_pos, target_pt, num_samples)


                    ray_origins = np.array([[p[0], 500.0, p[2]] for p in samples])
                    ray_directions = np.tile([0.0, -1.0, 0.0], (num_samples, 1))



                    hits = intersector.intersects_any(ray_origins=ray_origins, ray_directions=ray_directions)


                    los_ratio = np.sum(hits) / float(num_samples)
                # ========================================================


                score = (los_ratio * 5000.0) - diff

                if score > best_score:
                    best_score = score
                    best_node_idx = reachable_map[idx_local]
                    final_safe_pt = nearest_nav_node_pt


            if best_node_idx == -1:
                dists_to_target = np.linalg.norm(reachable_centers[:, [0, 2]] - target_pt[[0, 2]], axis=1)
                fallback_idx_local = np.argmin(dists_to_target)

                best_node_idx = reachable_map[fallback_idx_local]
                final_safe_pt = reachable_centers[fallback_idx_local]


            raw_path = []
            curr = best_node_idx
            while curr is not None:
                raw_path.append(nav_graph.centers[curr])
                curr = predecessors[curr]
            raw_path.reverse()

            pruned_path = [start_pt]
            for pt in raw_path:
                vec_t_pt = (pt - target_pt);
                vec_t_pt[1] = 0
                if np.dot(vec_t_start, vec_t_pt) < -1e-4:
                    break

                if len(pruned_path) > 1:
                    if np.dot(pt - pruned_path[-1], final_safe_pt - pt) < 0:
                        break

                pruned_path.append(pt)

            if np.linalg.norm(pruned_path[-1][[0, 2]] - final_safe_pt[[0, 2]]) > 0.1:
                vec_t_final = (final_safe_pt - target_pt);
                vec_t_final[1] = 0
                if np.dot(vec_t_start, vec_t_final) >= -1e-4:
                    pruned_path.append(final_safe_pt)


            path_np = np.array(pruned_path)
            if len(path_np) > 1:
                mask = np.linalg.norm(path_np[1:][:, [0, 2]] - path_np[:-1][:, [0, 2]], axis=1) > 1e-3
                path_np = np.vstack([path_np[0], path_np[1:][mask]])

            cleaned_path = self._clean_path_for_spline(path_np)
            optimized_path = self.fix_start_direction(start_pt, cleaned_path)
            target_paths.append(np.array(optimized_path))

        return target_paths, actual_radii

    def get_surround_paths_to_targets(self, nav_graph, start_face_idx, target_points_yup, distances, predecessors,
                                      target_scales=None, radius_threshold=None, safety_margin_factor=1.0):
        target_paths = []
        target_radius = []

        start_pt = getattr(self, 'agent_pos', np.array([0.0, 0.0, 0.0]))
        all_centers = nav_graph.centers

        if hasattr(nav_graph, 'tree'):
            tree = nav_graph.tree
        else:
            tree = cKDTree(all_centers)

        if hasattr(nav_graph, 'tree_xz'):
            tree_xz = nav_graph.tree_xz
        else:
            tree_xz = cKDTree(all_centers[:, [0, 2]])

        intersector = None
        if hasattr(nav_graph, 'mesh'):
            try:
                import trimesh
                intersector = trimesh.ray.ray_triangle.RayMeshIntersector(nav_graph.mesh)
            except Exception as e:
                print(f"Warning: Failed to initialize RayMeshIntersector: {e}")

        def dist_xz(p1, p2):
            return np.linalg.norm(p1[[0, 2]] - p2[[0, 2]])

        for i, target_pt in enumerate(target_points_yup):

            obj_scale = target_scales[i] if (target_scales is not None and i < len(target_scales)) else 0.5
            dist_agent_target = dist_xz(start_pt, target_pt)
            radius = min(1.0 * dist_agent_target, 2.0 * obj_scale)
            dist_diff = dist_agent_target - radius
            if dist_diff > radius_threshold:
                print(f"radius: {radius}, dist_agent_target: {dist_agent_target}, dist_agent_target - radius:"
                      f" {dist_diff} > radius_threshold: {radius_threshold}, increase radius to {dist_agent_target - radius_threshold}.")
                radius = dist_agent_target - radius_threshold
            else:
                print(f"radius: {radius}, dist_agent_target: {dist_agent_target}, dist_agent_target - radius:"
                      f" {dist_diff} <= radius_threshold: {radius_threshold}.")
            current_radius = radius
            num_samples = 72
            angles = np.linspace(0, 360, num_samples, endpoint=False)

            max_jump_dist = (radius * 2 * np.pi / num_samples) * 10.0


            valid_points_map = {}
            agent_y = all_centers[start_face_idx][1]  # 用地板高度，而非 agent 原始高度

            for ang_idx, ang in enumerate(angles):
                theta = np.radians(ang)
                query_pos = np.array([
                    target_pt[0] + radius * np.cos(theta),
                    agent_y,
                    target_pt[2] + radius * np.sin(theta)
                ])

                if intersector is not None:
                    ray_origin = np.array([[query_pos[0], 500.0, query_pos[2]]])
                    ray_direction = np.array([[0.0, -1.0, 0.0]])
                    if not intersector.intersects_any(ray_origins=ray_origin, ray_directions=ray_direction)[0]:
                        continue

                query_xz = np.array([query_pos[0], query_pos[2]])
                dists, idxs = tree_xz.query(query_xz, k=10)
                if isinstance(idxs, (int, np.integer)): idxs = [idxs]

                for node_idx in idxs:
                    if node_idx == start_face_idx or node_idx in predecessors:
                        node_pos = all_centers[node_idx]
                        # height filtering
                        if abs(node_pos[1] - agent_y) > 0.3:
                            continue
                        valid_points_map[ang_idx] = {
                            'node_idx': node_idx,
                            'pos': node_pos,
                            'global_dist': distances[node_idx]
                        }
                        break

            if not valid_points_map:
                d, idx = tree.query(target_pt, k=1)
                target_paths.append([start_pt, all_centers[idx]])
                target_radius.append(current_radius)
                continue


            initial_split_ang_idx = min(valid_points_map, key=lambda k: valid_points_map[k]['global_dist'])



            sorted_samples = []
            for ang_idx, data in valid_points_map.items():
                diff = (ang_idx - initial_split_ang_idx + num_samples) % num_samples
                if diff > num_samples / 2: diff -= num_samples
                sorted_samples.append({'diff': diff, 'node_idx': data['node_idx'], 'pos': data['pos']})

            sorted_samples.sort(key=lambda x: x['diff'])



            split_idx_in_sorted = next(i for i, s in enumerate(sorted_samples) if s['diff'] == 0)


            valid_arc_cw = [sorted_samples[split_idx_in_sorted]]
            for i in range(split_idx_in_sorted + 1, len(sorted_samples)):
                if np.linalg.norm(valid_arc_cw[-1]['pos'] - sorted_samples[i]['pos']) < max_jump_dist:
                    valid_arc_cw.append(sorted_samples[i])
                else:
                    break


            valid_arc_ccw = []
            for i in range(split_idx_in_sorted - 1, -1, -1):

                ref_pos = valid_arc_ccw[0]['pos'] if valid_arc_ccw else sorted_samples[split_idx_in_sorted]['pos']
                if np.linalg.norm(ref_pos - sorted_samples[i]['pos']) < max_jump_dist:
                    valid_arc_ccw.insert(0, sorted_samples[i])
                else:
                    break

            combined_valid_arc = valid_arc_ccw + valid_arc_cw
            limit_steps = num_samples // 2

            # max_step_range = num_samples // 2
            # arc_node_indices = [s['node_idx'] for s in combined_valid_arc if abs(s['diff']) <= max_step_range]

            if len(combined_valid_arc) > limit_steps:

                idx_in_arc = len(valid_arc_ccw)



                start_idx = max(0, idx_in_arc - limit_steps // 2)
                end_idx = min(len(combined_valid_arc), start_idx + limit_steps)


                if end_idx - start_idx < limit_steps:
                    start_idx = max(0, end_idx - limit_steps)

                final_arc_samples = combined_valid_arc[start_idx:end_idx]
            else:
                final_arc_samples = combined_valid_arc

            arc_node_indices = [s['node_idx'] for s in final_arc_samples]


            tip_a, tip_b = arc_node_indices[0], arc_node_indices[-1]
            dist_a = distances.get(tip_a, float('inf'))
            dist_b = distances.get(tip_b, float('inf'))
            if dist_a <= dist_b:
                full_arc_indices = arc_node_indices
            else:
                full_arc_indices = arc_node_indices[::-1]

            full_arc_points = [all_centers[idx] for idx in full_arc_indices]

            arc_entry_pt = all_centers[full_arc_indices[0]]
            start_node_pt = all_centers[start_face_idx]

            curr = full_arc_indices[0]
            temp_path_indices = []
            path_found = False
            while curr is not None and curr != -9999:
                temp_path_indices.append(curr)
                if curr == start_face_idx:
                    path_found = True
                    break
                curr = predecessors.get(curr)

            if path_found:
                temp_path_indices.reverse()
                path_a_points = [all_centers[idx] for idx in temp_path_indices]
            else:
                path_a_points = [start_node_pt, arc_entry_pt]

            if np.linalg.norm(path_a_points[0] - start_pt) > 0.01:
                path_a_points.insert(0, start_pt)


            pruned_a = []
            for pt in path_a_points[:-1]:
                if dist_xz(pt, target_pt) < current_radius * 0.98: continue
                pruned_a.append(pt)

            full_raw_path = pruned_a + full_arc_points
            clean_path = self._clean_path_for_spline(full_raw_path)

            target_radius.append(current_radius)
            target_paths.append(clean_path)

        return target_paths, target_radius

    def _clean_path_for_spline(self, path_points):
        if path_points is None or len(path_points) == 0:
            return None

        path_arr = np.array(path_points)

        cleaned = []
        threshold = self.spacing

        for p in path_arr:
            if len(cleaned) == 0:
                cleaned.append(p)
                continue

            dists = np.linalg.norm(np.array(cleaned) - p, axis=1)

            if np.min(dists) > threshold:
                cleaned.append(p)

        cleaned = np.array(cleaned)

        count = len(cleaned)
        if count < 4:
            if count == 0: return None
            if count == 1: return np.tile(cleaned, (4, 1))


            dists = np.linalg.norm(cleaned[1:] - cleaned[:-1], axis=1)
            cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
            total_len = cum_dist[-1]

            if total_len < 1e-6:

                return np.tile(cleaned[0], (4, 1))


            target_dists = np.linspace(0, total_len, 5)


            new_x = np.interp(target_dists, cum_dist, cleaned[:, 0])
            new_y = np.interp(target_dists, cum_dist, cleaned[:, 1])
            new_z = np.interp(target_dists, cum_dist, cleaned[:, 2])

            cleaned = np.stack([new_x, new_y, new_z], axis=1)

        return cleaned

    def explore_agents(self, start_face_idx: int, num_directions=8) -> List[List[np.ndarray]]:
        if start_face_idx == -1: return []
        distances = {node: float('inf') for node in range(len(self.centers))}
        distances[start_face_idx] = 0
        predecessors = {start_face_idx: None}
        pq = [(0, start_face_idx)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > distances[u]: continue
            for v, weight in self.adjacency[u]:
                new_dist = d + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(pq, (new_dist, v))

        start_pt = np.array([0.0, 0.0, 0.0])
        sector_best = {}

        for face_idx, center in enumerate(self.centers):
            if distances[face_idx] == float('inf'): continue
            if face_idx == start_face_idx: continue

            dx = center[0] - start_pt[0]
            dz = center[2] - start_pt[2]
            angle = math.atan2(dz, dx)
            if angle < 0: angle += 2 * math.pi
            sector_step = (2 * math.pi) / num_directions
            sector_id = int(angle / sector_step)

            displacement = np.linalg.norm(center - start_pt)

            if sector_id not in sector_best or displacement > sector_best[sector_id][1]:
                sector_best[sector_id] = (face_idx, displacement)

        all_paths = []
        for sec_id, (end_node, disp) in sector_best.items():
            if disp < 0.2: continue
            raw_path_centers = []
            curr = end_node
            while curr is not None:
                raw_path_centers.append(self.centers[curr])
                curr = predecessors[curr]
            raw_path_centers.reverse()

            optimized_control_points = self.fix_start_direction(start_pt, raw_path_centers)
            all_paths.append(optimized_control_points)
        return all_paths

    def fix_start_direction(self, start_pt: np.ndarray, path_centers: List[np.ndarray]) -> List[np.ndarray]:
        """
        Prefix-suppression version: trim only the initial backward or sharply deviating points. Once the first forward point is found, preserve all following obstacle-avoidance nodes.
        """
        if path_centers is None or len(path_centers) < 2:
            return [start_pt] + (path_centers if path_centers else [])

        points = np.array(path_centers)
        start_xy = np.array([start_pt[0], start_pt[2]])
        point_xys = np.stack([points[:, 0], points[:, 2]], axis=1)



        base_vec_xy = point_xys[-1] - start_xy
        base_norm = np.linalg.norm(base_vec_xy)

        if base_norm < 1e-6:
            return [start_pt] + path_centers


        vecs_xy = point_xys - start_xy
        vec_norms = np.linalg.norm(vecs_xy, axis=1)
        vec_norms[vec_norms < 1e-6] = 1e-6

        dots = np.sum(vecs_xy * base_vec_xy, axis=1)
        cos_thetas = dots / (vec_norms * base_norm)
        cos_thetas = np.clip(cos_thetas, -1.0, 1.0)


        threshold = 0.5


        first_valid_idx = 0
        for i, cos_val in enumerate(cos_thetas):

            if cos_val > threshold and vec_norms[i] > 0.05:
                first_valid_idx = i
                break
        else:

            first_valid_idx = len(points) - 1


        return [start_pt] + list(points[first_valid_idx:])


def get_max_size_center(mask_np, depth_map, rays, std_threshold=5.):
    """
    Compute the maximum cross-section size and center point of an object in 3D space.

    Updated logic:
    Use standard-deviation detection. If the depth variance inside the mask is large, which may indicate background contamination, project with the minimum-like 5th percentile depth; otherwise use median depth.
    """


    if isinstance(mask_np, torch.Tensor):
        mask_np = mask_np.detach().cpu().numpy()

    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()

    H, W = depth_map.shape


    y_indices, x_indices = np.where(mask_np > 0)
    if len(y_indices) == 0:
        return 0.0, np.zeros(3), False

    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)


    cx_2d = (min_x + max_x) / 2.0
    cy_2d = (min_y + max_y) / 2.0


    mask_depths = depth_map[y_indices, x_indices]
    valid_depths = mask_depths[mask_depths > 0.01]

    if len(valid_depths) == 0:
        return 0.0, np.zeros(3), False


    d_std = np.std(valid_depths)


    if d_std > std_threshold:
        final_depth = np.percentile(valid_depths, 5)
    else:
        final_depth = np.median(valid_depths)


    def internal_project(x, y, z_val):
        xi = int(np.clip(x, 0, W - 1))
        yi = int(np.clip(y, 0, H - 1))

        if isinstance(rays, torch.Tensor):
            ray = rays[yi, xi].detach().cpu().numpy()
        else:
            ray = rays[yi, xi]

        return ray * z_val


    center_3d = internal_project(cx_2d, cy_2d, final_depth)


    p_left = internal_project(min_x, cy_2d, final_depth)
    p_right = internal_project(max_x, cy_2d, final_depth)
    width_3d = np.linalg.norm(p_left - p_right)


    p_top = internal_project(cx_2d, min_y, final_depth)
    p_bottom = internal_project(cx_2d, max_y, final_depth)
    height_3d = np.linalg.norm(p_top - p_bottom)


    max_size = min(width_3d, height_3d)

    # max_size = max(width_3d, height_3d)

    return max_size, center_3d, True


def project_center_to_3d(center_2d, depth_map, rays, mask, std_threshold=0.5):
    """
    Project a 2D center point into 3D using the mask and depth map.

    Args:
        center_2d: Object 2D center point as (cx, cy).
        depth_map: Depth map with shape (H, W).
        rays: Ray directions with shape (H, W, 3).
        mask: Object binary mask with shape (H, W), where 1 marks object pixels.
        std_threshold: Threshold for detecting an abnormal depth distribution in meters. If the standard deviation inside the mask exceeds this value, the mask may include background, so use the minimum-depth strategy.

    Returns:
        point_3d: [x, y, z]
        depth_val: float
    """

    cx, cy = center_2d


    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    if isinstance(rays, torch.Tensor):
        rays = rays.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    H, W = depth_map.shape
    cx_int = int(np.clip(cx, 0, W - 1))
    cy_int = int(np.clip(cy, 0, H - 1))



    mask_bool = (mask > 0)


    masked_depths = depth_map[mask_bool]


    valid_depths = masked_depths[masked_depths > 0.001]


    if len(valid_depths) == 0:

        final_depth = depth_map[cy_int, cx_int]

        if final_depth <= 0.001: final_depth = 0.0
    else:

        depth_std = np.std(valid_depths)





        if depth_std > std_threshold:

            final_depth = np.percentile(valid_depths, 5)

        else:

            final_depth = np.median(valid_depths)



    ray = rays[cy_int, cx_int]


    point_3d = ray * final_depth

    return point_3d.tolist(), float(final_depth)


def find_robust_center(mask_np, depth_map):
    """
    Compute a robust center point for a mask.

    Steps:
    1. Extract all depth values inside the mask.
    2. Remove abnormal depth values by keeping pixels between the 25th and 75th percentiles, filtering edge noise and depth discontinuities.
    3. Compute the spatial centroid of the remaining pixels.
    4. Snap the centroid to the nearest valid pixel so the returned point is inside the mask and has valid depth.
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    depth_map = np.array(depth_map)


    y_indices, x_indices = np.where(mask_np > 0)
    if len(y_indices) == 0:
        return None


    depths = depth_map[y_indices, x_indices]



    valid_mask = depths > 1e-3


    if np.sum(valid_mask) == 0:
        valid_mask = np.ones_like(depths, dtype=bool)

    valid_depths = depths[valid_mask]
    valid_y = y_indices[valid_mask]
    valid_x = x_indices[valid_mask]




    q25 = np.percentile(valid_depths, 25)
    q75 = np.percentile(valid_depths, 75)


    iqr_mask = (valid_depths >= q25) & (valid_depths <= q75)


    if np.sum(iqr_mask) > 0:
        final_y = valid_y[iqr_mask]
        final_x = valid_x[iqr_mask]
    else:
        final_y = valid_y
        final_x = valid_x



    centroid_x = np.mean(final_x)
    centroid_y = np.mean(final_y)




    dist_sq = (final_x - centroid_x) ** 2 + (final_y - centroid_y) ** 2
    best_idx = np.argmin(dist_sq)

    center_x = float(final_x[best_idx])
    center_y = float(final_y[best_idx])

    return [center_x, center_y]


def connect_navmesh_components(verts, faces, max_bridge_z_diff=5.0):
    """
    V4.1: Add a height-difference constraint on top of the original winding-order fix. This prevents incorrectly connecting floors near Z~0 to tabletops near Z~0.7.
    """
    if not faces:
        return verts, faces

    verts_np = np.array(verts)
    faces_np = np.array(faces)


    adj = collections.defaultdict(list)
    vert_to_face_indices = collections.defaultdict(list)
    edge_counts = collections.defaultdict(int)
    directed_edge_to_face = {}

    for idx, face in enumerate(faces_np):
        for v_idx in face:
            vert_to_face_indices[v_idx].append(idx)
        edges_undirected = [tuple(sorted((face[0], face[1]))), tuple(sorted((face[1], face[2]))), tuple(sorted((face[2], face[0])))]
        for edge in edges_undirected:
            edge_counts[edge] += 1
        directed_edge_to_face[(face[0], face[1])] = idx
        directed_edge_to_face[(face[1], face[2])] = idx
        directed_edge_to_face[(face[2], face[0])] = idx

    boundary_verts_set = set()
    for edge, count in edge_counts.items():
        if count == 1:
            boundary_verts_set.add(edge[0])
            boundary_verts_set.add(edge[1])


    edge_to_face = collections.defaultdict(list)
    for idx, face in enumerate(faces_np):
        edges = [tuple(sorted((face[0], face[1]))), tuple(sorted((face[1], face[2]))), tuple(sorted((face[2], face[0])))]
        for edge in edges:
            edge_to_face[edge].append(idx)
    face_adj = collections.defaultdict(list)
    for edge, f_indices in edge_to_face.items():
        for i in range(len(f_indices)):
            for j in range(i + 1, len(f_indices)):
                u, v = f_indices[i], f_indices[j]
                face_adj[u].append(v)
                face_adj[v].append(u)
    visited = set()
    components = []
    for i in range(len(faces_np)):
        if i in visited: continue
        component = []
        queue = collections.deque([i])
        visited.add(i); component.append(i)
        while queue:
            curr = queue.popleft()
            for neighbor in face_adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor); component.append(neighbor); queue.append(neighbor)
        components.append(component)

    if len(components) <= 1:
        return verts, faces


    components.sort(key=len, reverse=True)
    main_island_face_indices = set(components[0])
    main_island_vert_indices = {v for f_idx in main_island_face_indices for v in faces_np[f_idx]}

    new_faces = list(faces)

    def get_oriented_bridge_face(v_connect, v_target, island_face_indices):
        my_faces = vert_to_face_indices[v_connect]
        for f_idx in my_faces:
            if f_idx in island_face_indices:
                face = faces_np[f_idx]
                for neighbor in face:
                    if neighbor == v_connect: continue
                    if edge_counts[tuple(sorted((v_connect, neighbor)))] != 1: continue
                    if (v_connect, neighbor) in directed_edge_to_face:
                        return (neighbor, v_connect, v_target)
                    elif (neighbor, v_connect) in directed_edge_to_face:
                        return (v_connect, neighbor, v_target)
        return None

    for i in range(1, len(components)):
        sub_island_face_indices = components[i]
        sub_island_vert_indices = {v for f_idx in sub_island_face_indices for v in faces_np[f_idx]}

        main_boundary_candidates = list(main_island_vert_indices.intersection(boundary_verts_set))
        sub_boundary_candidates = list(sub_island_vert_indices.intersection(boundary_verts_set))

        if not main_boundary_candidates or not sub_boundary_candidates:
            continue


        tree = cKDTree(verts_np[main_boundary_candidates])
        dists, indices = tree.query(verts_np[sub_boundary_candidates], k=1)
        min_idx = np.argmin(dists)

        closest_sub_v = sub_boundary_candidates[min_idx]
        closest_main_v = main_boundary_candidates[indices[min_idx]]


        z_diff = abs(verts_np[closest_sub_v][2] - verts_np[closest_main_v][2])
        if z_diff > max_bridge_z_diff:
            print(f" {z_diff:.4f}m > {max_bridge_z_diff}m continue")
            continue
        # -----------------------------

        bridge_face_1 = get_oriented_bridge_face(closest_main_v, closest_sub_v, main_island_face_indices)
        bridge_face_2 = get_oriented_bridge_face(closest_sub_v, closest_main_v, set(sub_island_face_indices))

        if bridge_face_1: new_faces.append(bridge_face_1)
        if bridge_face_2: new_faces.append(bridge_face_2)

        main_island_face_indices.update(sub_island_face_indices)
        main_island_vert_indices.update(sub_island_vert_indices)

    return verts, new_faces


def get_reconstruct_destinations(mesh, segmentation_data, nav_verts=None,
                                 R_to_yup=None, num_reconstruct=5,
                                 is_outdoor=False, global_median_depth=20.0):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0: return [], []

    R_inv = R_to_yup.T if R_to_yup is not None else np.eye(3)


    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    e = [np.linalg.norm(v1 - v0, axis=1), np.linalg.norm(v2 - v1, axis=1), np.linalg.norm(v0 - v2, axis=1)]
    max_edge_per_face = np.max(e, axis=0)
    s = sum(e) / 2.0
    area_per_face = np.sqrt(np.clip(s * (s - e[0]) * (s - e[1]) * (s - e[2]), 0, None))
    aspect_ratios = max_edge_per_face / (np.min(e, axis=0) + 1e-6)

    face_centers = (v0 + v1 + v2) / 3.0
    face_centers_zup = face_centers @ R_inv.T
    face_dists_h = np.linalg.norm(face_centers_zup[:, :2], axis=1)


    adaptive_stretch_thresholds = 6.0 + (face_dists_h / (global_median_depth + 1e-6))
    stretched_mask = aspect_ratios > adaptive_stretch_thresholds

    stretched_indices = np.where(stretched_mask)[0]
    if len(stretched_indices) == 0: return [], []


    stretched_submesh = trimesh.Trimesh(vertices=vertices, faces=triangles[stretched_indices], process=False)
    face_groups = trimesh.graph.connected_components(stretched_submesh.face_adjacency, nodes=np.arange(len(stretched_submesh.faces)))

    raw_clusters = []
    for group in face_groups:
        orig_idx = stretched_indices[group]
        raw_clusters.append({
            "indices": orig_idx,
            "area": np.sum(area_per_face[orig_idx]),
            "center": np.mean(face_centers[orig_idx], axis=0),
            "max_edge": np.max(max_edge_per_face[orig_idx]),
            "dist_h": np.mean(face_dists_h[orig_idx])
        })


    raw_clusters = [c for c in raw_clusters if c['area'] > 0.01]
    raw_clusters.sort(key=lambda x: -x['area'])
    merged_clusters = []
    while len(raw_clusters) > 0:
        base = raw_clusters.pop(0)
        to_merge_idx = [i for i, c in enumerate(raw_clusters) if np.linalg.norm(base['center'] - c['center']) < 0.6]
        for i in reversed(to_merge_idx):
            m = raw_clusters.pop(i)
            base['center'] = (base['center'] * base['area'] + m['center'] * m['area']) / (base['area'] + m['area'])
            base['area'] += m['area']
            base['max_edge'] = max(base['max_edge'], m['max_edge'])
        merged_clusters.append(base)

    # =========================================================

    vis_all_candidates = []
    vis_status_map = {}
    # =========================================================


    object_buckets = {}
    void_candidates = []

    for cluster in merged_clusters:
        center_yup = cluster['center']
        center_zup = R_inv @ center_yup
        dist_h = np.linalg.norm(center_zup[:2])

        if is_outdoor and dist_h > 8 * global_median_depth:

            dummy_cand = {"target_pt_zup": center_zup, "scale": 0.0}
            vis_all_candidates.append(dummy_cand)
            vis_status_map[id(dummy_cand)] = "filtered_by_outdoor"
            continue

        adaptive_factor = (dist_h / (global_median_depth + 1e-6))
        adaptive_factor = np.clip(adaptive_factor, 0, 2)
        effective_scale = cluster['max_edge'] * adaptive_factor

        best_obj, min_dist_obj = None, float('inf')
        for obj in segmentation_data:
            dist = np.linalg.norm(center_zup - np.array(obj['center_point_3d']))
            if dist < min_dist_obj:
                min_dist_obj, best_obj = dist, obj

        cand_data = {
            "area": cluster['area'],
            "target_pt_zup": center_zup,
            "scale": effective_scale,
            "dist_h": dist_h,
            "max_edge": cluster['max_edge']
        }

        vis_all_candidates.append(cand_data)

        if best_obj and min_dist_obj < 3.0:
            oid = best_obj['id']
            cand_data.update({"obj": best_obj, "label": best_obj.get("label", "unknown")})
            if oid not in object_buckets: object_buckets[oid] = []
            object_buckets[oid].append(cand_data)
        else:
            cand_data.update({"obj": {"id": -1, "label": "structural_void"}, "label": "void_area"})
            void_candidates.append(cand_data)


    all_potential = []
    for oid, group in object_buckets.items():
        group.sort(key=lambda x: -x['area'])
        all_potential.extend(group[:4])

        for dropped in group[4:]:
            vis_status_map[id(dropped)] = "filtered_by_bucket"

    all_potential.extend(void_candidates)
    all_potential.sort(key=lambda x: -x['area'])


    for cand in all_potential:
        vis_status_map[id(cand)] = "filtered_by_capacity"

    final_selected = []
    is_kept_by_nms = [False] * len(all_potential)
    for idx, cand in enumerate(all_potential):
        is_redundant = False
        for accepted in final_selected:
            dist = np.linalg.norm(cand['target_pt_zup'] - accepted['target_pt_zup'])
            nms_radius = max(min(cand['scale'], accepted['scale']), 1.0)

            if dist < nms_radius:
                is_redundant = True
                break

        if not is_redundant:
            final_selected.append(cand)
            is_kept_by_nms[idx] = True
        else:
            vis_status_map[id(cand)] = "filtered_by_nms"

        if len(final_selected) >= num_reconstruct:
            break


    if len(final_selected) < num_reconstruct:
        for idx, cand in enumerate(all_potential):
            if not is_kept_by_nms[idx] and len(final_selected) < num_reconstruct:
                final_selected.append(cand)

    # =========================================================

    # =========================================================
    if len(vis_all_candidates) > 0:
        combined_mesh = o3d.geometry.TriangleMesh()
        base_sphere_radius = 0.1

        actual_returned = final_selected[:num_reconstruct]
        for c in actual_returned:
            vis_status_map[id(c)] = "selected"

        for cand in vis_all_candidates:
            pt_zup = cand['target_pt_zup']
            status = vis_status_map.get(id(cand), "unknown")
            cand_scale = cand.get('scale', 0.0)


            if status == "selected":
                color = [0.2, 0.8, 0.2]
                radius = base_sphere_radius
            elif status == "filtered_by_nms":
                color = [0.8, 0.2, 0.2]
                radius = base_sphere_radius
            elif status == "filtered_by_capacity":
                color = [0.2, 0.5, 0.8]
                radius = base_sphere_radius
            elif status == "filtered_by_bucket":
                color = [0.9, 0.6, 0.1]
                radius = base_sphere_radius
            elif status == "filtered_by_outdoor":
                color = [0.6, 0.2, 0.8]
                radius = base_sphere_radius
            else:
                color = [0.5, 0.5, 0.5]
                radius = base_sphere_radius


            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(pt_zup)
            sphere.paint_uniform_color(color)
            sphere.compute_vertex_normals()
            combined_mesh += sphere


            if cand_scale > 0.05:

                torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=cand_scale, tube_radius=0.05)
                torus.translate(pt_zup)
                torus.paint_uniform_color(color)
                torus.compute_vertex_normals()
                combined_mesh += torus

    # =========================================================
    return final_selected[:num_reconstruct], final_selected


# =================================================================================
# Part 2: Visualization Helpers (Walls, Cubes, Maps)
# =================================================================================

def save_recon_markers_only(all_candidates, top5_info, save_path):
    """
    Precise visualization:
    - Blue: all detected stretched islands in the scene as background reference.
    - Red: the top 5 reconstruction target points as exactly 5 spheres.
    """
    all_vertices = []
    all_colors = []



    blue_color = [0, 0, 255, 255]
    for cand in all_candidates:

        marker_positions = cand.get('stretched_candidates', [cand['target_pt_zup']])
        for pos in marker_positions:
            sphere = trimesh.creation.icosphere(radius=0.06, subdivisions=2)
            sphere.apply_translation(pos)


            points, _ = trimesh.sample.sample_surface(sphere, count=500)

            all_vertices.append(points)
            all_colors.append(np.tile(blue_color, (points.shape[0], 1)).astype(np.uint8))


    red_color = [255, 0, 0, 255]
    for item in top5_info:
        pos = item['target_pt']
        sphere = trimesh.creation.icosphere(radius=0.15, subdivisions=2)
        sphere.apply_translation(pos)

        points, _ = trimesh.sample.sample_surface(sphere, count=1500)

        all_vertices.append(points)
        all_colors.append(np.tile(red_color, (points.shape[0], 1)).astype(np.uint8))

    if not all_vertices:
        print("Warning: No reconstruction markers found.")
        return


    combined_pcd = trimesh.PointCloud(vertices=np.vstack(all_vertices), colors=np.vstack(all_colors))
    combined_pcd.export(save_path)
    print(f"Successfully saved markers. (Red: {len(top5_info)} | Blue: {len(all_candidates)} candidates)")


def add_cube_marker(center, size, color, verts_list, colors_list, faces_list):
    """
    Create a cube at the specified location.
    """
    half = size / 2.0
    x, y, z = center
    start_idx = len(verts_list)

    # 8 vertices
    verts_list.append([x - half, y - half, z - half])
    verts_list.append([x + half, y - half, z - half])
    verts_list.append([x + half, y + half, z - half])
    verts_list.append([x - half, y + half, z - half])
    verts_list.append([x - half, y - half, z + half])
    verts_list.append([x + half, y - half, z + half])
    verts_list.append([x + half, y + half, z + half])
    verts_list.append([x - half, y + half, z + half])

    for _ in range(8):
        colors_list.append(color)

    # 12 Faces
    faces_list.append([start_idx + 0, start_idx + 2, start_idx + 1])
    faces_list.append([start_idx + 0, start_idx + 3, start_idx + 2])
    faces_list.append([start_idx + 4, start_idx + 5, start_idx + 6])
    faces_list.append([start_idx + 4, start_idx + 6, start_idx + 7])
    faces_list.append([start_idx + 0, start_idx + 1, start_idx + 5])
    faces_list.append([start_idx + 0, start_idx + 5, start_idx + 4])
    faces_list.append([start_idx + 1, start_idx + 2, start_idx + 6])
    faces_list.append([start_idx + 1, start_idx + 6, start_idx + 5])
    faces_list.append([start_idx + 2, start_idx + 3, start_idx + 7])
    faces_list.append([start_idx + 2, start_idx + 7, start_idx + 6])
    faces_list.append([start_idx + 3, start_idx + 0, start_idx + 4])
    faces_list.append([start_idx + 3, start_idx + 4, start_idx + 7])


def write_scene_with_paths(filename, base_verts, base_faces, base_color, paths, wall_height=0.2, marker_size=0.15):
    print(f"Generating PLY: {filename} ...")

    all_verts = []
    all_colors = []
    all_faces = []

    use_single_color = True
    if hasattr(base_color, '__len__') and len(base_color) == len(base_verts) and len(base_verts) > 3:
        use_single_color = False

    for i, v in enumerate(base_verts):
        all_verts.append(v)
        if use_single_color:
            all_colors.append(base_color)
        else:
            all_colors.append(base_color[i])

    for f in base_faces:
        all_faces.append(f)

    current_idx = len(all_verts)

    path_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
    ]
    start_marker_color = (255, 255, 255)
    end_marker_color = (255, 0, 0)

    for i, path in enumerate(paths):
        c = path_colors[i % len(path_colors)]
        if len(path) < 1: continue

        add_cube_marker(path[0], marker_size, start_marker_color, all_verts, all_colors, all_faces)
        current_idx = len(all_verts)

        if len(path) > 1:
            add_cube_marker(path[-1], marker_size, end_marker_color, all_verts, all_colors, all_faces)
            current_idx = len(all_verts)

        if len(path) < 2: continue
        for k in range(len(path) - 1):
            p1 = path[k]
            p2 = path[k + 1]

            v_bl = np.array([p1[0], p1[1], p1[2]])
            v_br = np.array([p2[0], p2[1], p2[2]])
            v_tl = np.array([p1[0], p1[1], p1[2] + wall_height])
            v_tr = np.array([p2[0], p2[1], p2[2] + wall_height])

            all_verts.extend([v_bl, v_br, v_tr, v_tl])
            all_colors.extend([c, c, c, c])

            all_faces.append((current_idx, current_idx + 1, current_idx + 2))
            all_faces.append((current_idx, current_idx + 2, current_idx + 3))
            all_faces.append((current_idx, current_idx + 2, current_idx + 1))
            all_faces.append((current_idx, current_idx + 3, current_idx + 2))

            current_idx += 4

    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(all_verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(all_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v, c in zip(all_verts, all_colors):
            r, g, b = int(c[0]), int(c[1]), int(c[2])
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {r} {g} {b}\n")

        for face in all_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Saved: {filename}")


def generate_centered_map(filename, nav_verts, nav_faces, paths,
                          img_size=1024,
                          target_points=None,
                          target_radii=None):
    """
    Generate a top-down map.

    Optimized behavior: target points, observation radii, paths, and endpoints are forced to share the same color, with different shapes used to distinguish ownership.
    """
    if cv2 is None:
        print("OpenCV not installed, skipping map generation.")
        return


    nav_xs = nav_verts[:, 0]
    nav_ys = nav_verts[:, 1]

    min_x, max_x = np.min(nav_xs), np.max(nav_xs)
    min_y, max_y = np.min(nav_ys), np.max(nav_ys)


    if paths:
        for path in paths:
            if path is None or len(path) == 0: continue
            p_xs = path[:, 0]
            p_ys = path[:, 1]
            min_x = min(min_x, np.min(p_xs))
            max_x = max(max_x, np.max(p_xs))
            min_y = min(min_y, np.min(p_ys))
            max_y = max(max_y, np.max(p_ys))


    if target_points is not None:
        t_xs = target_points[:, 0]
        t_ys = target_points[:, 1]
        min_x = min(min_x, np.min(t_xs))
        max_x = max(max_x, np.max(t_xs))
        min_y = min(min_y, np.min(t_ys))
        max_y = max(max_y, np.max(t_ys))

    scene_w = max_x - min_x
    scene_h = max_y - min_y

    if scene_w < 1e-3: scene_w = 1.0
    if scene_h < 1e-3: scene_h = 1.0


    margin_ratio = 0.1
    draw_w = img_size * (1 - margin_ratio)
    draw_h = img_size * (1 - margin_ratio)

    scale_x = draw_w / scene_w
    scale_y = draw_h / scene_h
    scale = min(scale_x, scale_y)

    final_scene_w = scene_w * scale
    final_scene_h = scene_h * scale
    offset_x = (img_size - final_scene_w) / 2.0
    offset_y = (img_size - final_scene_h) / 2.0

    def to_px(x, y):
        px = int((x - min_x) * scale + offset_x)
        py = int((max_y - y) * scale + offset_y)
        return px, py


    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)


    gray_color = (60, 60, 60)
    nav_px_coords = [to_px(v[0], v[1]) for v in nav_verts]

    for face in nav_faces:
        pts = []
        for idx in face:
            pts.append(nav_px_coords[idx])
        pts_np = np.array([pts], dtype=np.int32)
        cv2.fillPoly(img, pts_np, gray_color)


    path_colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 100, 50),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 255, 128),
        (128, 255, 0),
        (200, 200, 200)
    ]

    for i, path in enumerate(paths):

        c = path_colors[i % len(path_colors)]


        if target_points is not None and i < len(target_points):
            tx, ty = target_points[i][0], target_points[i][1]
            t_px, t_py = to_px(tx, ty)


            cv2.drawMarker(img, (t_px, t_py), c, markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=14, thickness=2)


            if target_radii is not None and i < len(target_radii):
                radius_meter = target_radii[i]
                radius_px = int(radius_meter * scale)
                if radius_px > 0:
                    cv2.circle(img, (t_px, t_py), radius_px, c, 1)


        if path is None or len(path) == 0: continue

        pts = []
        for p in path:
            pts.append(to_px(p[0], p[1]))

        if len(pts) > 0:
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts)], False, c, thickness=2)


            cv2.circle(img, pts[0], 4, (255, 255, 255), -1)


            cv2.circle(img, pts[-1], 6, c, -1)
            cv2.circle(img, pts[-1], 2, (255, 255, 255), -1)

    cv2.imwrite(filename, img)
    print(f"Saved centered map to {filename}")


# =================================================================================
# Part 3: Original Batch Logic (Modified)
# =================================================================================

def flatten_verts(verts: List[Tuple[float, float, float]]) -> List[float]:
    out = []
    for (x, y, z) in verts:
        out.extend((float(x), float(y), float(z)))
    return out


def flatten_indices(faces: List[Tuple[int, int, int]]) -> List[int]:
    out = []
    for (a, b, c) in faces:
        out.extend((int(a), int(b), int(c)))
    return out


def filter_navmesh_by_height(verts, faces, max_height):
    valid_faces = []
    for face in faces:
        y0 = verts[face[0]][1]
        y1 = verts[face[1]][1]
        y2 = verts[face[2]][1]
        avg_y = (y0 + y1 + y2) / 3.0
        if avg_y < max_height:
            valid_faces.append(face)
    return valid_faces


def build_navmesh_from_mesh(verts, faces, cellSize=0.05, cellHeight=0.05,
                            agentHeight=1.6, agentRadius=0.3,
                            agentMaxClimb=0.4, maxSlope=45.0):
    if recast is None:
        print("recast module not available; skipping navmesh build.")
        return None, None, "no_recast"
    try:
        rc = recast.RecastNavMesh()
    except Exception:
        return None, None, "rc_construct_failed"

    v_flat = flatten_verts(verts)
    i_flat = flatten_indices(faces)

    try:
        ok = rc.build_from_vertices(v_flat, i_flat, cellSize, cellHeight,
                                    agentHeight, agentRadius, agentMaxClimb, maxSlope)
    except TypeError:
        ok = False
        try:
            ok = rc.build_from_vertices(v_flat, i_flat, cellSize, cellHeight)
        except Exception:
            ok = False
    except Exception:
        ok = False

    try:
        pv_flat, pt_flat = rc.get_polymesh()
        try:
            pv_flat = list(pv_flat)
        except:
            pass
        try:
            pt_flat = list(pt_flat)
        except:
            pass
        return pv_flat, pt_flat, rc
    except Exception as e:
        print("rc.get_polymesh() failed:", e)
        return None, None, rc


def filter_largest_navmesh_component(verts, faces):
    """
    Keep the connected component with the most faces in the NavMesh and remove unused vertices.
    """
    if not faces:
        return verts, faces


    verts_np = np.array(verts)
    faces_np = np.array(faces)
    num_faces = len(faces_np)



    adj = collections.defaultdict(list)
    edge_to_face = collections.defaultdict(list)


    for idx, face in enumerate(faces_np):

        edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0])))
        ]
        for edge in edges:
            edge_to_face[edge].append(idx)


    for edge, f_indices in edge_to_face.items():

        for i in range(len(f_indices)):
            for j in range(i + 1, len(f_indices)):
                u, v = f_indices[i], f_indices[j]
                adj[u].append(v)
                adj[v].append(u)


    visited = set()
    components = []

    for i in range(num_faces):
        if i in visited:
            continue


        component = []
        queue = collections.deque([i])
        visited.add(i)
        component.append(i)

        while queue:
            curr = queue.popleft()
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        components.append(component)

    if not components:
        return [], []


    largest_component_indices = max(components, key=len)
    print(f"NavMesh Filter: Kept largest island with {len(largest_component_indices)} faces (dropped {num_faces - len(largest_component_indices)} faces).")



    kept_faces = faces_np[largest_component_indices]


    unique_vert_indices = np.unique(kept_faces)


    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vert_indices)}


    new_verts = verts_np[unique_vert_indices]



    mapper = np.vectorize(old_to_new.get)
    new_faces = mapper(kept_faces)


    return [tuple(v) for v in new_verts], [tuple(f) for f in new_faces]


# -------------------------
# Save artifacts (Modified to include Paths)
# -------------------------
def save_artifacts(output_dir: str, mesh, nav_verts=None, nav_faces=None, exploration_paths=None,
                   target_paths=None, surround_paths=None, reconstruct_paths=None,
                   target_points=None, target_radius=None, nav_node_centers=None,
                   reconstruct_targets=None, reconstruct_radius=None):
    """
    Save the mesh, NavMesh, and four path categories.

    Includes a Top 5 logic optimization specifically for reconstruction tasks.
    """
    os.makedirs(output_dir, exist_ok=True)
    R_back = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))


    mesh.rotate(R_back, center=(0, 0, 0))
    mesh_min_bound = mesh.get_min_bound()
    mesh_max_bound = mesh.get_max_bound()
    mesh_verts_rotated = np.asarray(mesh.vertices)
    mesh_faces_rotated = np.asarray(mesh.triangles)

    try:
        mesh_colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else (200, 200, 200)
    except:
        mesh_colors = (200, 200, 200)

    # o3d.io.write_triangle_mesh(os.path.join(output_dir, "mesh_reconstructed_save.ply"), mesh, compressed=True)


    if nav_verts is not None and nav_faces is not None:
        try:
            nav_verts_rotated = np.array(nav_verts) @ R_back.T


            if nav_node_centers is not None:
                plt.figure(figsize=(10, 10))
                plt.triplot(nav_verts_rotated[:, 0], nav_verts_rotated[:, 1], nav_faces, color='gray', alpha=0.3, lw=0.5)
                centers_rot = np.array(nav_node_centers) @ R_back.T
                plt.scatter(centers_rot[:, 0], centers_rot[:, 1], c='red', s=1, alpha=0.5)
                plt.axis('equal')
                plt.savefig(os.path.join(output_dir, "navmesh_sampling_view.png"), bbox_inches='tight')
                plt.close()


            def rotate_path(p):
                return np.array(p) @ R_back.T if p is not None else None

            tasks = [
                ("exploration", [rotate_path(p) for p in (exploration_paths or [])]),
                ("target", [rotate_path(p) for p in (target_paths or [])]),
                ("surround", [rotate_path(p) for p in (surround_paths or [])]),
                ("reconstruct", [rotate_path(p) for p in (reconstruct_paths or [])]),
            ]


            for task_name, paths in tasks:
                if not paths: continue
                sub_dir = os.path.join(output_dir, task_name)
                os.makedirs(sub_dir, exist_ok=True)


                with open(os.path.join(sub_dir, "paths.json"), "w") as w:
                    json.dump([p.tolist() if p is not None else None for p in paths], w, indent=2)


                if 'write_scene_with_paths' in globals():
                    write_scene_with_paths(
                        os.path.join(sub_dir, "vis_mesh.ply"),
                        mesh_verts_rotated, mesh_faces_rotated, mesh_colors,
                        [p for p in paths if p is not None], wall_height=0.2
                    )


                if 'generate_centered_map' in globals():
                    map_target_pts = None
                    map_target_rad = None


                    if task_name == "surround" and target_points is not None:
                        map_target_pts = target_points @ R_back.T
                        map_target_rad = target_radius


                    elif task_name == "reconstruct" and reconstruct_targets is not None:
                        map_target_pts = np.array(reconstruct_targets) @ R_back.T
                        map_target_rad = reconstruct_radius

                    generate_centered_map(
                        os.path.join(sub_dir, "map.png"),
                        nav_verts_rotated, nav_faces,
                        [p for p in paths if p is not None],
                        target_points=map_target_pts,
                        target_radii=map_target_rad
                    )

            # --- 5. Metadata ---
            metadata = {
                "bbox": {"min": mesh_min_bound.tolist(), "max": mesh_max_bound.tolist()},
                "num_reconstruct_tasks": len(reconstruct_paths) if reconstruct_paths else 0
            }
            with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            print(f"Artifact saving failed: {e}")
            traceback.print_exc()


def get_font(image_width):
    font_size = max(20, int(image_width / 60))
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue
    if font is None:
        font = ImageFont.load_default()
    return font


def calculate_mask_center(mask_np):
    y_indices, x_indices = np.where(mask_np > 0)
    if len(y_indices) == 0: return None
    return [float(np.mean(x_indices)), float(np.mean(y_indices))]


def save_visualization(image_pil, masks_np, labels, directions, output_path):
    """
    Updated visualization: add direction information to labels and avoid overlap with spiral search placement.
    """
    orig_w, orig_h = image_pil.size
    combined_overlay = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
    base_rgba = image_pil.convert("RGBA")
    draw_overlay = ImageDraw.Draw(combined_overlay)
    font = get_font(orig_w)
    random_seed_base = 42


    for i, mask in enumerate(masks_np):
        mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        if mask_pil.size != (orig_w, orig_h):
            mask_pil = mask_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
        binary_mask = np.array(mask_pil) > 128

        rnd = random.Random(random_seed_base + i)
        color = (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255))

        rgba_arr = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        rgba_arr[..., 0] = color[0]
        rgba_arr[..., 1] = color[1]
        rgba_arr[..., 2] = color[2]
        rgba_arr[..., 3] = (binary_mask.astype(np.uint8) * 140)
        combined_overlay = Image.alpha_composite(combined_overlay, Image.fromarray(rgba_arr, mode="RGBA"))


    draw_text = ImageDraw.Draw(combined_overlay)
    drawn_boxes = []

    for i, mask in enumerate(masks_np):
        mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        if mask_pil.size != (orig_w, orig_h):
            mask_pil = mask_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
        binary_mask = np.array(mask_pil) > 128

        center = calculate_mask_center(binary_mask)
        if center:
            cx, cy = center
            label_text = f"{labels[i]} ({directions[i]})"


            r = max(3, int(orig_w / 500))
            draw_text.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 0, 0, 255), outline=(255, 255, 255, 255), width=2)


            try:

                bbox = draw_text.textbbox((0, 0), label_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except:
                text_w, text_h = draw_text.textsize(label_text, font=font)

            pad = 10
            box_w = text_w + pad * 2
            box_h = text_h + pad * 2


            angle = 0.0
            radius = 0.0
            step_angle = 0.5
            step_radius = 2.0

            new_cx, new_cy = cx, cy
            rect_x0, rect_y0, rect_x1, rect_y1 = 0, 0, 0, 0

            def _clamp_label_rect(ncx, ncy):
                rx0 = ncx - box_w / 2
                ry0 = ncy - box_h / 2
                rx1 = ncx + box_w / 2
                ry1 = ncy + box_h / 2
                ncx_, ncy_ = float(ncx), float(ncy)
                if rx0 < 0:
                    rx0, rx1 = 0, box_w
                    ncx_ = box_w / 2
                if ry0 < 0:
                    ry0, ry1 = 0, box_h
                    ncy_ = box_h / 2
                if rx1 > orig_w:
                    rx0, rx1 = orig_w - box_w, orig_w
                    ncx_ = orig_w - box_w / 2
                if ry1 > orig_h:
                    ry0, ry1 = orig_h - box_h, orig_h
                    ncy_ = orig_h - box_h / 2
                return ncx_, ncy_, rx0, ry0, rx1, ry1


            max_spiral_steps = max(8000, (orig_w + orig_h) * 6)
            placed = False
            for _ in range(max_spiral_steps):
                new_cx = cx + radius * math.cos(angle)
                new_cy = cy + radius * math.sin(angle)
                new_cx, new_cy, rect_x0, rect_y0, rect_x1, rect_y1 = _clamp_label_rect(new_cx, new_cy)

                overlap = False
                margin = 4
                for bx0, by0, bx1, by1 in drawn_boxes:
                    if not (rect_x1 + margin < bx0 or rect_x0 - margin > bx1 or
                            rect_y1 + margin < by0 or rect_y0 - margin > by1):
                        overlap = True
                        break

                if not overlap:
                    placed = True
                    break

                angle += step_angle
                radius += step_radius

            if not placed:
                new_cx, new_cy, rect_x0, rect_y0, rect_x1, rect_y1 = _clamp_label_rect(cx, cy)
            # ----------------------------------------


            drawn_boxes.append((rect_x0, rect_y0, rect_x1, rect_y1))


            if math.hypot(new_cx - cx, new_cy - cy) > 10:
                draw_text.line([(cx, cy), (new_cx, new_cy)], fill=(255, 255, 255, 180), width=2)


            draw_text.rectangle((rect_x0, rect_y0, rect_x1, rect_y1), fill=(0, 0, 0, 200), outline=(255, 255, 255, 128), width=2)

            draw_text.text((rect_x0 + pad, rect_y0 + pad), label_text, font=font, fill=(255, 255, 255, 255))

    final_image = Image.alpha_composite(base_rgba, combined_overlay)
    final_image.convert("RGB").save(output_path)

def select_reconstruct_via_fps(processed_seg_list, top_k):
    """
    Weighted farthest-point sampling for reconstruction tasks.
    Score = sqrt(Area) * Distance_to_Selected_Set
    """
    if len(processed_seg_list) <= top_k:
        return processed_seg_list, list(range(len(processed_seg_list)))


    points = np.array([item.get("target_pt") for item in processed_seg_list])

    areas = np.array([item.get("area", 1.0) for item in processed_seg_list])

    selected_indices = []


    first_idx = np.argmax(areas)
    selected_indices.append(first_idx)

    remaining_indices = list(set(range(len(processed_seg_list))) - set(selected_indices))


    while len(selected_indices) < top_k and remaining_indices:
        max_score = -1
        best_next_idx = -1

        for idx in remaining_indices:

            dists = np.linalg.norm(points[selected_indices] - points[idx], axis=1)
            min_dist = np.min(dists)


            score = np.sqrt(areas[idx]) * min_dist

            if score > max_score:
                max_score = score
                best_next_idx = idx

        if best_next_idx != -1:
            selected_indices.append(best_next_idx)
            remaining_indices.remove(best_next_idx)
        else:
            break


    final_list = [processed_seg_list[i] for i in selected_indices]
    return final_list, selected_indices


def filter_and_select_diverse_trajectories(
        trajectories,
        origin_trajs,
        start_point,
        max_k=4,

        overlap_ratio=0.5,
        dist_threshold=0.1,
        sample_points=50,

        min_angle_thresh=np.pi / 4,
        min_length_ratio=0.2,
        length_weight=0.3,
        diversity_weight=0.7,
):
    """
    Two-stage filtering:
      Stage 1 - Deduplication: remove geometrically overlapping redundant trajectories, keeping the longer ones.
      Stage 2 - Diversity selection: greedily choose up to max_k trajectories from the deduplicated set, balancing length and angular diversity.

    Args:
        trajectories  : list[np.ndarray] - N trajectories, each shaped (M_i, 3), used for geometric comparison.
        origin_trajs  : list[np.ndarray] - N original trajectories, used to compute true length ordering.
        start_point   : np.ndarray (2,) or (3,) - Start coordinate; the first two dimensions are used for angle computation.
        max_k         : int - Maximum number of trajectories to select.
        overlap_ratio : float - Similarity coefficient; checks whether the first x% of a short trajectory overlaps with a long trajectory.
        dist_threshold: float - Maximum average error for treating trajectories as overlapping.
        sample_points : int - Number of sampled points used for comparison.
        min_angle_thresh : float - Minimum angle difference from already selected trajectories, in radians.
        min_length_ratio : float - Filter trajectories shorter than longest trajectory * ratio.
        length_weight    : float - Weight of length in the combined score.
        diversity_weight : float - Weight of angular diversity in the combined score.

    Returns:
        selected_indices : list[int] - Indices of the selected trajectories in the original list.
    """

    n = len(trajectories)
    if n == 0:
        return []

    # =================================================================

    # =================================================================
    start_2d = np.array(start_point[:2], dtype=float)

    traj_info = []
    for idx, path in enumerate(trajectories):
        info = {'idx': idx}


        if len(path) < 2:
            info.update({'len': 0, 'cum_dist': np.array([0.0]), 'path': path})
        else:
            diffs = np.linalg.norm(path[1:] - path[:-1], axis=1)
            cum_dist = np.insert(np.cumsum(diffs), 0, 0.0)
            info.update({'len': cum_dist[-1], 'cum_dist': cum_dist, 'path': path})


        path_origin = origin_trajs[idx]
        if len(path_origin) < 2:
            info['origin_len'] = 0
        else:
            diffs_o = np.linalg.norm(path_origin[1:] - path_origin[:-1], axis=1)
            info['origin_len'] = float(np.sum(diffs_o))


        endpoint_2d = np.array(path[-1][:2], dtype=float)
        direction = endpoint_2d - start_2d
        info['angle'] = float(np.arctan2(direction[1], direction[0]))

        traj_info.append(info)

    # =================================================================

    # =================================================================


    sorted_info = sorted(traj_info, key=lambda x: x['origin_len'], reverse=True)

    dedup_indices = []
    dedup_trajs = []

    for curr in sorted_info:
        if curr['len'] == 0:
            continue

        is_redundant = False

        for ref in dedup_trajs:
            check_len = curr['len'] * overlap_ratio
            if check_len > ref['len']:
                continue

            try:
                target_dists = np.linspace(0, check_len, sample_points)
                f_curr = interp1d(curr['cum_dist'], curr['path'], axis=0, assume_sorted=True)
                pts_curr = f_curr(target_dists)
                f_ref = interp1d(ref['cum_dist'], ref['path'], axis=0, assume_sorted=True)
                pts_ref = f_ref(target_dists)
                mean_dist = np.mean(np.linalg.norm(pts_curr - pts_ref, axis=1))

                if mean_dist < dist_threshold:
                    is_redundant = True
                    break
            except Exception:
                continue

        if not is_redundant:
            dedup_indices.append(curr['idx'])
            dedup_trajs.append(curr)

    if not dedup_indices:
        return []

    # =================================================================

    # =================================================================


    dedup_info_map = {t['idx']: t for t in dedup_trajs}
    candidates = [dedup_info_map[i] for i in dedup_indices]


    max_origin_len = max(c['origin_len'] for c in candidates)
    if max_origin_len == 0:
        return []
    candidates = [c for c in candidates if c['origin_len'] >= max_origin_len * min_length_ratio]

    if not candidates:
        return []


    for c in candidates:
        c['norm_len'] = c['origin_len'] / max_origin_len

    def angular_distance(a1, a2):
        """
        Minimum angular distance between two angles in [0, pi].
        """
        diff = abs(a1 - a2)
        return min(diff, 2 * np.pi - diff)

    selected = []


    first = max(candidates, key=lambda c: c['origin_len'])
    selected.append(first)
    remaining = [c for c in candidates if c['idx'] != first['idx']]


    while len(selected) < max_k and remaining:
        best_candidate = None
        best_score = -1

        for cand in remaining:

            min_ang_dist = min(angular_distance(cand['angle'], s['angle']) for s in selected)


            if min_ang_dist < min_angle_thresh:
                continue


            score = (length_weight * cand['norm_len']
                     + diversity_weight * (min_ang_dist / np.pi))

            if score > best_score:
                best_score = score
                best_candidate = cand


        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining = [c for c in remaining if c['idx'] != best_candidate['idx']]

    selected_indices = [s['idx'] for s in selected]
    return selected_indices


def process_trajectories(trajectories, D, X, smoothing=0.1, world_up=np.array([0, 0, 1]), look_at_target=None, is_recon=False):
    """
    Process trajectories with B-Spline fitting, path truncation, uniform sampling, and camera pose construction.

    For recon tasks, apply special filtering: truncate or drop severe backward motion (>120 degrees) and drop excessive pitch angles (>45 degrees).

    Args:
        trajectories: Input trajectory list, each trajectory shaped (M, 3).
        D: Maximum path length; truncate fitted curves longer than D.
        X: Number of uniformly sampled output points.
        smoothing: B-Spline smoothing parameter.
        world_up: World-space up direction with shape (3,).
        look_at_target: Optional (3,) np.array; if provided, the camera always looks at this target.
        is_recon: bool. If True, enable strict view-angle and backward-motion filtering for reconstruction.

    Returns:
        all_poses: Camera pose array with shape (N, X, 4, 4).
    """
    N = len(trajectories)
    all_poses = []

    for i in range(N):
        traj = trajectories[i]
        M = traj.shape[0]


        try:
            weights = np.ones(M)
            weights[0] = 1e5
            tck, u = splprep(traj.T, s=smoothing, k=min(3, M - 1), w=weights)
        except Exception as e:
            print(f"轨迹 {i} 拟合失败: {e}")
            continue


        dense_samples = 1000
        u_dense = np.linspace(0, 1, dense_samples)
        xyz_dense = np.array(splev(u_dense, tck)).T


        start_offset = traj[0] - xyz_dense[0]
        xyz_dense += start_offset


        if is_recon and look_at_target is not None:
            move_vecs = np.diff(xyz_dense, axis=0)
            look_vecs = look_at_target.reshape(1, 3) - xyz_dense[:-1]

            move_norms = np.linalg.norm(move_vecs, axis=1, keepdims=True)
            move_norms[move_norms < 1e-6] = 1e-6
            move_vecs_norm = move_vecs / move_norms

            look_norms = np.linalg.norm(look_vecs, axis=1, keepdims=True)
            look_norms[look_norms < 1e-6] = 1e-6
            look_vecs_norm = look_vecs / look_norms

            cos_theta = np.sum(move_vecs_norm * look_vecs_norm, axis=1)
            backward_idx = np.where(cos_theta < -0.501)[0]

            if len(backward_idx) > 0:
                cut_idx = backward_idx[0]
                if cut_idx < (dense_samples * 0.3):
                    continue
                xyz_dense = xyz_dense[:cut_idx + 1]
        # ===============================================================


        diffs = np.linalg.norm(xyz_dense[1:] - xyz_dense[:-1], axis=1)
        cum_dist = np.cumsum(diffs)
        cum_dist = np.insert(cum_dist, 0, 0.0)
        total_len = cum_dist[-1]


        if D is not None and D > 0 and total_len > D:
            effective_len = D
            print(f"Trajectory: {total_len:.3f} is longer than {D:.3f}, clip to {D:.3f}")
        else:
            effective_len = total_len


        target_dists = np.linspace(0, effective_len, X)

        final_x = np.interp(target_dists, cum_dist, xyz_dense[:, 0])
        final_y = np.interp(target_dists, cum_dist, xyz_dense[:, 1])
        final_z = np.interp(target_dists, cum_dist, xyz_dense[:, 2])

        points = np.stack([final_x, final_y, final_z], axis=1)  # (X, 3)


        if look_at_target is not None:

            z_axis_raw = look_at_target.reshape(1, 3) - points
            z_norms = np.linalg.norm(z_axis_raw, axis=1, keepdims=True)
            z_norms[z_norms < 1e-6] = 1e-6
            z_axis = z_axis_raw / z_norms
        else:

            tangents = np.gradient(points, axis=0)
            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1e-6
            z_axis = tangents / norms


        if is_recon:
            up_norm = world_up / np.linalg.norm(world_up)
            pitch_sines = np.abs(np.dot(z_axis, up_norm))
            # sin(45°) ≈ 0.7071
            if np.any(pitch_sines > 0.7071):
                continue
        # ===============================================================


        poses = np.eye(4).reshape(1, 4, 4).repeat(X, axis=0)
        poses[:, :3, 3] = points


        x_axis_raw = np.cross(z_axis, world_up)
        x_norms = np.linalg.norm(x_axis_raw, axis=1, keepdims=True)


        mask_singular = x_norms.flatten() < 1e-6
        x_axis_raw[mask_singular] = np.array([1, 0, 0])
        x_norms[mask_singular] = 1.0
        x_axis = x_axis_raw / x_norms


        y_axis = np.cross(z_axis, x_axis)

        poses[:, :3, 0] = x_axis
        poses[:, :3, 1] = y_axis
        poses[:, :3, 2] = z_axis

        all_poses.append(poses)
    return np.array(all_poses)


def visualize_comparison(original, processed, idx=0, save_path=None):
    """
    Visualize the comparison between the original and processed trajectories.

    Includes two figures:
    1. Trajectory in the original world coordinate system.
    2. Trajectory after rotation alignment, with the starting direction facing the positive Y axis.

    Arrows are drawn on the curve to show motion direction.
    """


    def add_arrows(ax, points, color='r', num_arrows=4):
        """
        Draw evenly spaced arrows along the trajectory.
        """
        if len(points) < 2: return


        span_x = np.max(points[:, 0]) - np.min(points[:, 0])
        span_y = np.max(points[:, 1]) - np.min(points[:, 1])
        span = max(span_x, span_y, 1e-3)


        head_width = span * 0.03
        head_length = span * 0.04


        indices = np.linspace(0, len(points) - 2, num_arrows + 2).astype(int)[1:-1]

        for i in indices:


            next_i = min(i + 5, len(points) - 1)
            if next_i <= i: next_i = i + 1

            start = points[i]
            end = points[next_i]

            dx = end[0] - start[0]
            dy = end[1] - start[1]


            norm = np.sqrt(dx ** 2 + dy ** 2)
            if norm < 1e-6: continue




            ax.arrow(start[0], start[1],
                     dx / norm * span * 0.001, dy / norm * span * 0.001,
                     head_width=head_width,
                     head_length=head_length,
                     fc=color, ec=color,
                     length_includes_head=True,
                     zorder=10)


    def get_pos(data):
        if data is None or len(data) == 0:
            return np.zeros((0, 3))
        if data.ndim == 3 and data.shape[1:] == (4, 4):
            return data[:, :3, 3]
        return data[:, :3]

    p_orig = get_pos(original)
    p_proc = get_pos(processed)

    if len(p_proc) < 2:
        print("数据过短，无法绘图")
        return


    fig1 = plt.figure(figsize=(6, 6))
    ax1 = fig1.add_subplot()

    if len(p_orig) > 0:
        ax1.plot(p_orig[:, 0], p_orig[:, 1], 'b.-', alpha=0.3, label='Original', markersize=2)


    ax1.plot(p_proc[:, 0], p_proc[:, 1], 'r.-', label='Processed', markersize=2)


    add_arrows(ax1, p_proc, color='r', num_arrows=3)


    ax1.plot(p_proc[0, 0], p_proc[0, 1], 'go', label='Start', markersize=8)
    ax1.plot(p_proc[-1, 0], p_proc[-1, 1], 'k*', label='End', markersize=10)

    ax1.set_xlabel('World X')
    ax1.set_ylabel('World Y')
    ax1.set_title(f'Trajectory (World Frame) - Frame {idx}')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig1)
    else:
        plt.show()


    lookahead = min(5, len(p_proc) - 1)
    start_vec = p_proc[lookahead] - p_proc[0]

    current_angle = np.arctan2(start_vec[1], start_vec[0])
    target_angle = np.pi / 2
    rotation_angle = target_angle - current_angle

    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    R = np.array([[c, -s], [s, c]])

    def transform_points(points, origin, rot_matrix):
        if len(points) == 0: return points
        centered = points[:, :2] - origin[:2]
        rotated = centered @ rot_matrix.T
        return rotated

    origin_pt = p_proc[0]
    p_orig_rot = transform_points(p_orig, origin_pt, R)
    p_proc_rot = transform_points(p_proc, origin_pt, R)


    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot()

    if len(p_orig_rot) > 0:
        ax2.plot(p_orig_rot[:, 0], p_orig_rot[:, 1], 'b.-', alpha=0.3, label='Original', markersize=2)

    ax2.plot(p_proc_rot[:, 0], p_proc_rot[:, 1], 'r.-', label='Processed', markersize=2)


    add_arrows(ax2, p_proc_rot, color='r', num_arrows=3)


    ax2.plot(0, 0, 'go', label='Start', markersize=8)


    max_range = np.max(np.abs(p_proc_rot)) if len(p_proc_rot) > 0 else 1.0
    ax2.arrow(0, 0, 0, max_range * 0.15, head_width=max_range * 0.05, head_length=max_range * 0.05, fc='gray', ec='gray', alpha=0.3, width=max_range * 0.01)
    ax2.text(max_range * 0.02, max_range * 0.1, "Init Dir (+Y)", fontsize=8, color='gray')

    ax2.set_xlabel('Local X')
    ax2.set_ylabel('Local Y (Forward)')
    ax2.set_title(f'Aligned Trajectory (Start -> +Y)')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()


    if save_path:
        root, ext = os.path.splitext(save_path)
        rot_save_path = f"{root}_direction{ext}"
        plt.savefig(rot_save_path)
        print(f"Saved rotated visualization to: {rot_save_path}")
        plt.close(fig2)
    else:
        plt.show()


def compute_trajectory_similarity_matrix(trajectories,
                                         pos_scale=1.0,
                                         rot_scale_deg=10.0,
                                         weights=(0.75, 0.25)):
    """
    Compute the pairwise similarity matrix for N camera trajectories.

    Similarity formula: Score = 1 / (1 + Error / Scale)
    Final score = 0.75 * Pos_Score + 0.25 * Rot_Score

    Args:
        trajectories (np.ndarray): Camera trajectory data (c2w) with shape (N, V, 4, 4).
        pos_scale (float): Sensitivity factor for position error, in the same units as the trajectories, such as meters. When average position error equals this value, position similarity is 0.5. Recommended value is roughly one tenth of scene scale.
        rot_scale_deg (float): Sensitivity factor for rotation error, in degrees. When average rotation error equals this value, rotation similarity is 0.5.
        weights (tuple): (position_weight, rotation_weight), default (0.75, 0.25).

    Returns:
        sim_matrix (np.ndarray): Matrix with shape (N, N) and values in [0, 1]. sim_matrix[i, j] is the similarity between trajectories i and j.
    """
    N, V, _, _ = trajectories.shape
    w_pos, w_rot = weights


    # ---------------------------------------------------------

    trans = trajectories[:, :, :3, 3]


    rot_mats = trajectories[:, :, :3, :3]
    quats = R.from_matrix(rot_mats.reshape(-1, 3, 3)).as_quat().reshape(N, V, 4)


    sim_matrix = np.zeros((N, N), dtype=np.float32)


    # ---------------------------------------------------------


    for i in range(N):


        t_i = trans[i:i + 1, :, :]

        t_all = trans



        pos_errors = np.linalg.norm(t_all - t_i, axis=2).mean(axis=1)



        score_pos = 1.0 / (1.0 + pos_errors / pos_scale)



        q_i = quats[i:i + 1, :, :]

        q_all = quats


        # sum(a*b, axis=2)
        dot_products = np.sum(q_all * q_i, axis=2)


        dot_products = np.clip(np.abs(dot_products), -1.0, 1.0)


        angles_rad = 2 * np.arccos(dot_products)
        angles_deg = np.degrees(angles_rad)


        rot_errors = angles_deg.mean(axis=1)


        score_rot = 1.0 / (1.0 + rot_errors / rot_scale_deg)


        # (N,)
        final_score = w_pos * score_pos + w_rot * score_rot


        sim_matrix[i, :] = final_score

    return sim_matrix


def snap_navmesh_to_surface(nav_verts, tm_scene, offset=0.05, search_height=0.05, max_lift_dist=1.0):
    """
    Step 1: snap vertices upward only, without lowering them or checking normals. If a vertex is below the ground, force it up to ground + offset.
    """
    if not nav_verts: return nav_verts

    nav_verts_arr = np.array(nav_verts)


    ray_origins = nav_verts_arr.copy()
    ray_origins[:, 1] += search_height

    ray_directions = np.zeros_like(ray_origins)
    ray_directions[:, 1] = -1.0

    try:

        locations, index_ray, _ = tm_scene.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )
    except Exception:
        return nav_verts


    new_nav_verts = nav_verts_arr.copy()

    for i, ray_idx in enumerate(index_ray):
        hit_y = locations[i][1]
        current_y = nav_verts_arr[ray_idx][1]



        if hit_y > current_y:



            if (hit_y - current_y) > max_lift_dist:
                continue


            new_nav_verts[ray_idx][1] = hit_y + offset

    return new_nav_verts.tolist()


def resolve_face_clipping_sampling(nav_verts, nav_faces, tm_scene, offset=0.05, search_height=0.05, samples_per_line=3):
    """
    Step 2: prevent face clipping by sampling faces, ignoring normals and checking height only. Sample points on each triangle; if any sample is below the ground, compute the required lift and raise all three vertices of that triangle together.
    """
    if not nav_verts or not nav_faces:
        return nav_verts

    nav_verts_arr = np.array(nav_verts)
    faces_arr = np.array(nav_faces)
    num_faces = len(faces_arr)


    tri_verts = nav_verts_arr[faces_arr]  # (N, 3, 3)
    centroids = tri_verts.mean(axis=1)  # (N, 3)

    sample_points_list = [centroids]


    t_values = np.linspace(0, 1, samples_per_line + 2)[1:-1]
    for t in t_values:
        for v_idx in range(3):
            # P = Center * (1-t) + Vertex * t
            p = centroids * (1 - t) + tri_verts[:, v_idx, :] * t
            sample_points_list.append(p)


    all_samples = np.vstack(sample_points_list)


    ray_origins = all_samples.copy()
    ray_origins[:, 1] += search_height
    ray_directions = np.zeros_like(ray_origins)
    ray_directions[:, 1] = -1.0

    locations, index_ray, _ = tm_scene.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )



    face_lift_req = np.zeros(num_faces)

    for i, ray_idx in enumerate(index_ray):
        face_idx = ray_idx % num_faces

        hit_y = locations[i][1]
        sample_y = all_samples[ray_idx][1]



        if hit_y > sample_y:


            lift_needed = (hit_y - sample_y) + offset


            if lift_needed > face_lift_req[face_idx]:
                face_lift_req[face_idx] = lift_needed


    vertex_lift = np.zeros(len(nav_verts))


    nonzero_faces = np.where(face_lift_req > 0)[0]

    for f_idx in nonzero_faces:
        lift = face_lift_req[f_idx]
        v_indices = nav_faces[f_idx]



        vertex_lift[v_indices[0]] = max(vertex_lift[v_indices[0]], lift)
        vertex_lift[v_indices[1]] = max(vertex_lift[v_indices[1]], lift)
        vertex_lift[v_indices[2]] = max(vertex_lift[v_indices[2]], lift)


    count = np.count_nonzero(vertex_lift)
    if count > 0:
        print(f"  [Anti-Clip] Lifting {count} vertices (Force Lift Mode).")
        nav_verts_arr[:, 1] += vertex_lift

    return nav_verts_arr.tolist()


# ==============================================================================
#  Helper Functions
# ==============================================================================

def pil_image_to_base64(pil_img, img_format="PNG"):
    """
    Convert a PIL Image to a base64 string matching the output format of encode_image.
    :param pil_img: PIL Image object in memory.
    :param img_format: Image format such as JPEG, PNG, or BMP. Defaults to PNG; PNG is recommended for transparent images.
    :return: UTF-8 base64 string matching the original encode_image result format.
    """

    img_byte_arr = BytesIO()

    pil_img.save(
        img_byte_arr,
        format=img_format,
        quality=95,
        exif=b""
    )

    img_byte = img_byte_arr.getvalue()

    return base64.b64encode(img_byte).decode('utf-8')


def get_navigation_instruction(force_vlm=False):
    instruction = (
        "You are a navigation assistant for a **ground-based walking robot**. "
        "From this panoramic image, identify and extract distinct visual landmarks and obstacles that are crucial for navigation. "
        "Each element in the output shall be a short word or phrase, with no single element exceeding 4 words. "
        "Do NOT introduce adjectives; use only common nouns to describe objects in the image, and avoid repeatedly describing similar objects. "

        "\n\n**Inclusion Criteria (Include these):**\n"
        "1. **Natural Elements (Vertical orientation):** Vertically growing plants such as trees, tree trunks (not logs on the ground), and large bushes.\n"
        "2. **Man-made Structures:** Prominent architectural elements and fixtures like pillars (structural, standing upright), statues, fences, and sheds.\n"
        "3. **Furniture and Fixtures:** Furniture intended for sitting or placement like tables, chairs, benches, and lamps (exclude overhead fixtures).\n"
        "4. **Distinct Objects:** Prominent items such as entry doors for buildings or rooms (exclude cabinet doors), vehicles (ground-based, not toys), trash cans (standing waste bins), and large machines (operational, not toys).\n"

        "\n\n**Exclusion Criteria (Ignore these):**\n"
        "- **Flat Surfaces:** Areas like lawns, pavements, floors, etc., that do not present vertical obstacles.\n"
        "- **High Objects:** Items out of reach such as rooftops, ceiling fixtures, the sky, etc.\n"
        "- **Vague Background:** Distant scenery like mountains and horizon lines.\n"

        "\nOutput the result strictly as a JSON list of strings.\n"
    )

    if force_vlm:
        instruction += "- If you see an object but are unsure, INCLUDE it. Do not return an empty list."
    return instruction


def get_topk_seg_data(seg_data, topk):
    """
    Sort seg_data by the specified rule and select the top-k elements with the smallest combined rank.

    Args:
        seg_data (list): List of dicts, each containing depth_distance and mask_area fields.
        topk (int): Number of elements to select.

    Returns:
        list: Top-k elements with the smallest combined rank.
    """

    if not seg_data or topk <= 0:
        return [], None
    topk = min(topk, len(seg_data))


    sorted_by_depth = sorted(seg_data, key=lambda x: x['depth_distance'], reverse=True)

    depth_rank_map = {id(elem): idx + 1 for idx, elem in enumerate(sorted_by_depth)}



    sorted_by_mask = sorted(seg_data, key=lambda x: x['mask_area'], reverse=True)

    mask_rank_map = {id(elem): idx + 1 for idx, elem in enumerate(sorted_by_mask)}



    seg_with_score = []
    for elem in seg_data:
        elem_id = id(elem)
        # total_score = depth_rank_map[elem_id] + mask_rank_map[elem_id]
        total_score = mask_rank_map[elem_id]
        elem["total_rank"] = total_score
        elem["depth_rank"] = depth_rank_map[elem_id]
        elem["mask_rank"] = mask_rank_map[elem_id]
        seg_with_score.append({
            'element': elem,
            'total_score': total_score,
            'depth_rank': depth_rank_map[elem_id],
            'mask_rank': mask_rank_map[elem_id]
        })

    seg_with_idx = list(enumerate(seg_with_score))


    seg_sorted_with_idx = sorted(seg_with_idx, key=lambda x: x[1]['total_score'])

    seg_with_score_sorted = [item[1] for item in seg_sorted_with_idx]
    sorted_indices = [item[0] for item in seg_sorted_with_idx]


    sorted_indices = sorted_indices[:topk]
    topk_elements = [item['element'] for item in seg_with_score_sorted[:topk]]

    return topk_elements, sorted_indices


def deduplicate_ordered(items):
    seen = set()
    result = []
    BANNED = ["floor", "ceiling", "sky", "ground", "wall"]
    for item in items:
        if not isinstance(item, str): continue
        clean = item.strip()
        lower = clean.lower()
        if any(b in lower for b in BANNED) and "door" not in lower: continue
        if lower not in seen:
            seen.add(lower)
            result.append(clean)
    return result


def find_foreground_center(mask_np, depth_map):
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    depth_map = np.array(depth_map)

    y_indices, x_indices = np.where(mask_np > 0)
    if len(y_indices) == 0:
        return None

    depths = depth_map[y_indices, x_indices]
    median_depth = np.median(depths)
    closest_idx = np.argmin(np.abs(depths - median_depth))

    center_x = float(x_indices[closest_idx])
    center_y = float(y_indices[closest_idx])

    return [center_x, center_y]


def get_bearing_and_direction(center_x, image_width):
    bearing = (center_x / image_width) * 360 - 180
    if -45 <= bearing < 45:
        direction = "Front"
    elif 45 <= bearing < 135:
        direction = "Right"
    elif -135 <= bearing < -45:
        direction = "Left"
    else:
        direction = "Back"
    return direction, round(bearing, 2)


def project_point_to_3d(center_2d, depth_map, rays):
    cx, cy = center_2d
    H, W = depth_map.shape
    cx_int = int(np.clip(cx, 0, W - 1))
    cy_int = int(np.clip(cy, 0, H - 1))
    d = depth_map[cy_int, cx_int].item()
    ray = rays[cy_int, cx_int].cpu().numpy()
    point_3d = ray * d
    return point_3d.tolist(), d


def create_and_save_combined_pcd(original_pcd, seg_results, output_path):
    all_vertices = [original_pcd.vertices]
    all_colors = [original_pcd.colors]
    for i, obj in enumerate(seg_results):
        pos = obj['center_point_3d']
        label = obj['label'].lower()
        if "door" in label or "gate" in label or "exit" in label:
            color = [255, 0, 0, 255]
            radius = 0.25
        else:
            import colorsys
            rgb = colorsys.hsv_to_rgb(i / (len(seg_results) + 1), 1.0, 1.0)
            color = [int(c * 255) for c in rgb] + [255]
            radius = 0.15
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        sphere.apply_translation(pos)
        marker_points, _ = trimesh.sample.sample_surface(sphere, count=1000)
        marker_colors = np.tile(color, (marker_points.shape[0], 1)).astype(np.uint8)
        all_vertices.append(marker_points)
        all_colors.append(marker_colors)
    combined_vertices = np.vstack(all_vertices)
    combined_colors = np.vstack(all_colors)
    combined_pcd = trimesh.PointCloud(vertices=combined_vertices, colors=combined_colors)
    combined_pcd.export(output_path)


def get_mask_edge_points_3d(mask_np, depth_map, rays, sample_ratio=0.05, min_samples=5):
    y_indices, x_indices = np.where(mask_np > 0)
    if len(x_indices) == 0:
        return (None, None), (None, None)

    sorted_indices = np.argsort(x_indices)
    x_sorted = x_indices[sorted_indices]
    y_sorted = y_indices[sorted_indices]

    num_points = len(x_sorted)
    k = max(min_samples, int(num_points * sample_ratio))
    k = min(k, num_points // 2)
    if k == 0: k = 1

    lx_candidates = x_sorted[:k]
    ly_candidates = y_sorted[:k]
    rx_candidates = x_sorted[-k:]
    ry_candidates = y_sorted[-k:]

    def get_median_point_3d_and_2d(xs, ys):
        points_3d = []
        points_2d = []
        for x, y in zip(xs, ys):
            pt3d, _ = project_point_to_3d([x, y], depth_map, rays)
            if pt3d is not None and np.all(np.isfinite(pt3d)) and np.linalg.norm(pt3d) > 0.1:
                points_3d.append(pt3d)
                points_2d.append([x, y])

        if not points_3d:
            return None, None

        median_pt_3d = np.median(np.array(points_3d), axis=0)
        median_pt_2d = np.median(np.array(points_2d), axis=0)
        return median_pt_3d, median_pt_2d

    left_3d_np, left_2d_np = get_median_point_3d_and_2d(lx_candidates, ly_candidates)
    right_3d_np, right_2d_np = get_median_point_3d_and_2d(rx_candidates, ry_candidates)

    left_3d = left_3d_np.tolist() if isinstance(left_3d_np, np.ndarray) else None
    left_2d = left_2d_np.tolist() if isinstance(left_2d_np, np.ndarray) else None
    right_3d = right_3d_np.tolist() if isinstance(right_3d_np, np.ndarray) else None
    right_2d = right_2d_np.tolist() if isinstance(right_2d_np, np.ndarray) else None

    return (left_3d, left_2d), (right_3d, right_2d)


# ==============================================================================
#  NavMesh Processing
# ==============================================================================


def process_single_scene(scene_dir, scene_name, mesh, args, segmentation_data=None,
                         global_median_depth=None, is_outdoor=False, timer=None):
    output_dir = f"{scene_dir}/navmesh"
    os.makedirs(output_dir, exist_ok=True)

    if not mesh.has_vertices():
        print(f"[{scene_name}] Error: Mesh has no vertices.")
        return



    global_mesh_origin_down_distance = None
    try:
        raycasting_scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        raycasting_scene.add_triangles(mesh_t)

        ray_origins = np.array([
            [1e-3, 0.0, 0.0],
            [-1e-3, 0.0, 0.0],
            [0.0, 1e-3, 0.0],
            [0.0, -1e-3, 0.0],
            [-1e-3, 1e-3, 0.0],
            [1e-3, 1e-3, 0.0],
            [-1e-3, -1e-3, 0.0],
            [1e-3, -1e-3, 0.0],
        ], dtype=np.float32)
        ray_directions = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (ray_origins.shape[0], 1))
        rays = np.concatenate([ray_origins, ray_directions], axis=1)

        hit_result = raycasting_scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = hit_result["t_hit"].numpy()
        valid_hit_mask = np.isfinite(t_hit)

        rank0_log(f"[{scene_name}] global_mesh_origin_down hit ratio: ({np.sum(valid_hit_mask)}/{len(valid_hit_mask)})")

        if np.any(valid_hit_mask):
            global_mesh_origin_down_distance = float(np.mean(t_hit[valid_hit_mask]))

    except Exception as e:
        rank0_log(f"[{scene_name}] Warning: Failed to compute global mesh origin down distance: {e}")

    if global_mesh_origin_down_distance is not None and np.isfinite(global_mesh_origin_down_distance):
        rank0_log(f"[{scene_name}] Global mesh origin down distance: {global_mesh_origin_down_distance}")
        if is_outdoor:
            global_mesh_origin_down_distance = np.clip(global_mesh_origin_down_distance, 0.2, 0.75)
        else:
            global_mesh_origin_down_distance = np.clip(global_mesh_origin_down_distance, 0.28, 0.78)
        agentHeight = global_mesh_origin_down_distance * 1.2
        agentRadius = agentHeight * 0.4
        agentMaxClimb = agentHeight * 0.5
    else:
        rank0_log(f"[{scene_name}] Global mesh origin down distance is None or not finite, using default values.")
        agentHeight = args.agentHeight
        agentRadius = args.agentRadius
        agentMaxClimb = args.agentMaxClimb

    with timer.track("Get navmesh and post-processing"):

        # 1. Rotation (Z-Up -> Y-Up for Recast)

        R_to_yup = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
        mesh.rotate(R_to_yup, center=(0, 0, 0))

        verts = [(float(x), float(y), float(z)) for x, y, z in np.asarray(mesh.vertices)]
        faces = [(int(a), int(b), int(c)) for a, b, c in np.asarray(mesh.triangles)]
        faces_double = faces + [(a, c, b) for (a, b, c) in faces]

        # 2. Build NavMesh
        pv_flat, pt_flat, rc_obj = build_navmesh_from_mesh(verts, faces_double,
                                                           cellSize=args.cellSize,
                                                           cellHeight=args.cellHeight,
                                                           agentHeight=agentHeight,
                                                           agentRadius=agentRadius,
                                                           agentMaxClimb=agentMaxClimb,
                                                           maxSlope=args.maxSlope)

        nav_verts = []
        nav_faces = []

        if pv_flat is None:
            print(f"[{scene_name}] NavMesh build failed.")
        else:

            nav_verts = [(float(pv_flat[i]), float(pv_flat[i + 1]), float(pv_flat[i + 2])) for i in range(0, len(pv_flat), 3)]
            nav_faces = [(int(pt_flat[i]), int(pt_flat[i + 1]), int(pt_flat[i + 2])) for i in range(0, len(pt_flat), 3)]


            nav_faces = filter_navmesh_by_height(nav_verts, nav_faces, args.roof_height_threshold)


            nav_verts, nav_faces = connect_navmesh_components(nav_verts, nav_faces)
            nav_verts, nav_faces = filter_largest_navmesh_component(nav_verts, nav_faces)

            tm_scene = trimesh.Trimesh(vertices=verts, faces=faces)
            nav_verts = snap_navmesh_to_surface(nav_verts, tm_scene, offset=args.roof_height_threshold * 0.5, search_height=args.roof_height_threshold * 2., max_lift_dist=0.01)
            nav_verts = resolve_face_clipping_sampling(nav_verts, nav_faces, tm_scene, offset=args.roof_height_threshold * 0.5, search_height=args.roof_height_threshold * 2., samples_per_line=3)


        num_reconstruct = 10
        reconstruct_topk_info = []
        if segmentation_data and len(segmentation_data) > 0:
            reconstruct_topk_info, all_candidates = get_reconstruct_destinations(
                mesh,
                segmentation_data,
                nav_verts=nav_verts,
                R_to_yup=R_to_yup,
                num_reconstruct=num_reconstruct,
                is_outdoor=is_outdoor,
                global_median_depth=global_median_depth
            )


            topk_json_data = []
            for item in reconstruct_topk_info:
                item_ = item['obj']
                item_["target_pt"] = item['target_pt_zup'].tolist()
                topk_json_data.append(item_)


            os.makedirs(os.path.join(scene_dir, "navmesh"), exist_ok=True)
            with open(os.path.join(scene_dir, "navmesh/reconstruct_pairs.json"), 'w') as f:
                json.dump(topk_json_data, f, indent=4)


            combined_vis_path = os.path.join(scene_dir, "navmesh/vis_recon.ply")
            save_recon_markers_only(all_candidates, topk_json_data, combined_vis_path)
        # ============================================================

    # 3. Path Planning
    with timer.track("Navmesh Path Planning"):
        exploration_paths = []
        target_paths = []
        surround_paths = []
        reconstruct_paths = []
        nav_graph = None
        surround_radius = None
        target_points_yup = None
        recon_targets_yup = None
        actual_recon_radii = None

        if len(nav_faces) > 0:
            try:
                nav_graph = NavMeshGraph(nav_verts, nav_faces, sample_spacing=0.05)
                agent_pos = np.array([0.0, 0.0, 0.0])
                start_node_idx, start_pt = nav_graph.find_nearest_node(agent_pos)

                if start_node_idx != -1:
                    # A. Dijkstra
                    distances, predecessors = nav_graph.compute_dijkstra(start_node_idx)

                    # B. Exploration
                    exploration_paths = nav_graph.explore_agents(start_node_idx, num_directions=8)
                    # C. Targets & Surround & Reconstruct
                    if segmentation_data and len(segmentation_data) > 0:
                        # Target Paths
                        target_points_zup = np.array([obj['center_point_3d'] for obj in segmentation_data])
                        target_points_yup = (R_to_yup @ target_points_zup.T).T
                        target_paths, _ = nav_graph.get_paths_to_targets(
                            nav_graph, start_node_idx, target_points_yup, distances, predecessors
                        )
                        # Surround Path
                        target_scales = np.array([obj.get('scale_3d', 1.0) for obj in segmentation_data])
                        surround_paths, surround_radius = nav_graph.get_surround_paths_to_targets(
                            nav_graph,
                            start_node_idx,
                            target_points_yup,
                            distances,
                            predecessors,
                            target_scales=target_scales,
                            radius_threshold=global_median_depth * args.radius_threshold,
                        )

                    # Reconstruct Paths

                    if len(reconstruct_topk_info) > 0:
                        recon_targets_yup = (R_to_yup @ np.array([item['target_pt_zup'] for item in reconstruct_topk_info]).T).T
                        recon_scales = np.array([item['scale'] for item in reconstruct_topk_info])
                        reconstruct_paths, actual_recon_radii = nav_graph.get_paths_to_targets(
                            nav_graph, start_node_idx, recon_targets_yup, distances, predecessors,
                            target_scales=recon_scales
                        )
                    print(f"DEBUG: Found {len(reconstruct_topk_info)} recon targets, generated {len(reconstruct_paths)} paths.")
                else:
                    print(f"[{scene_name}] Warning: Start point not on NavMesh.")

            except Exception as e:
                rank0_log(f"[{scene_name}] Path planning failed: {e}", "ERROR")
                traceback.print_exc()

    # Save artifacts
    with timer.track("[IO] Save results of path planning from navmesh"):
        save_artifacts(output_dir, mesh, nav_verts, nav_faces,
                       exploration_paths=exploration_paths,
                       target_paths=target_paths,
                       surround_paths=surround_paths,
                       reconstruct_paths=reconstruct_paths,
                       target_points=target_points_yup,
                       target_radius=surround_radius,
                       nav_node_centers=nav_graph.centers,
                       reconstruct_targets=recon_targets_yup,
                       reconstruct_radius=actual_recon_radii)
    print(f"[{scene_name}] NavMesh processing complete.")