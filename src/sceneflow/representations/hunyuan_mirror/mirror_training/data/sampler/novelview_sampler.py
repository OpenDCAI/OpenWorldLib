import numpy as np
from typing import List, Tuple, Dict, Any
from mirror_training.utils.geometry import calculate_unprojected_mask


class ViewSamplingStrategy:
    def __init__(self, seed=None, video_aug=False):
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.video_aug = video_aug
        
    def compute_view_overlap(self, context_views: List[Dict], target_views: List[Dict]) -> float:
        """
        Compute the overlap between source views and novel views
        Evaluate based on camera position, orientation, and field of view overlap
        """
        if not context_views or not target_views:
            return 0.0
            
        unproj_mask = calculate_unprojected_mask(context_views, target_views)
        novel_valid_mask = np.concatenate([view["valid_mask"] for view in target_views], axis=0)
        total_overlap = unproj_mask.float().sum().item()/novel_valid_mask.sum()
            
        return total_overlap if target_views else 0.0
    
    def select_views_with_overlap_optimization(self, 
                                             all_cameras: List[Dict],
                                             source_count: int,
                                             num_candidates: int = 5) -> Tuple[List[int], List[int]]:
        """
        Select specific source and novel views with overlap optimization
        Use multi-candidate strategy to select the optimal combination
        """
        best_overlap = -1
        best_source_indices = None
        best_novel_indices = None
        
        # Generate multiple candidate combinations
        for _ in range(num_candidates):
            # Randomly select source views
            source_indices = self.rng.choice(len(all_cameras), size=source_count, replace=False).tolist()
            novel_indices = [i for i in range(len(all_cameras)) if i not in source_indices]
            
            # Calculate total overlap for current combination
            source_cameras = [all_cameras[i] for i in source_indices]
            novel_cameras = [all_cameras[i] for i in novel_indices]
            current_overlap = self.compute_view_overlap(source_cameras, novel_cameras)
            
            # Update best combination
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_source_indices = source_indices
                best_novel_indices = novel_indices
        
        return best_source_indices, best_novel_indices
    

    def select_views_with_bounded_video(
        self,
        all_cameras: List[Dict],
        source_count: int,
        num_candidates: int = 5,
    ) -> Tuple[List[int], List[int]]:
        """
        Sampling for video sequences:
        - Indices are arranged in consecutive frames
        - Novel indices must be within the [min(source), max(source)] range
        - Under constraint satisfaction, prioritize combinations with higher overlap
        """
        N = len(all_cameras)

        # Minimum required interval length (including endpoints)
        # Within the interval, must accommodate source_count + novel_count non-overlapping indices

        best_overlap = -1.0
        best_source_indices = None
        best_novel_indices = None

        for _ in range(num_candidates):
            # 1) Sample an interval [min_idx, max_idx] that satisfies the length requirement
            min_idx = 0
            max_idx = N - 1 

            # 2) Place endpoints as source boundaries within [min_idx, max_idx] 
            #    (ensuring novel views can definitely be within the interval)
            #    If source_count == 2, only use the endpoints; otherwise continue filling from within the interval
            source_indices_set = {min_idx, max_idx}
            # Randomly fill remaining source positions from available inner positions (excluding selected endpoints)
            if source_count > 2:
                inner_pool = [i for i in range(min_idx + 1, max_idx) if i not in source_indices_set]
                picked = self.rng.choice(inner_pool, size=(source_count - 2), replace=False).tolist()
                source_indices_set.update(picked)

            source_indices = sorted(source_indices_set)

            # 3) Prepare novel candidates: within interval and not overlapping with source
            novel_indices = [i for i in range(min_idx, max_idx + 1) if i not in source_indices_set]

            # 5) Calculate total overlap for the combination to compare multiple candidates
            source_cameras = [all_cameras[i] for i in source_indices]
            novel_cameras = [all_cameras[i] for i in novel_indices]
            current_overlap = self.compute_view_overlap(source_cameras, novel_cameras)

            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_source_indices = source_indices
                best_novel_indices = novel_indices

        return best_source_indices, best_novel_indices
    
    def sample_views(self, all_cameras: List[Dict], source_count: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Main sampling function: sample source and novel views according to strategy
        """
        if len(all_cameras) <= 2 or len(all_cameras) < source_count:
            for view in all_cameras:
                view["is_target"] = False
            return all_cameras, []

        novel_count = len(all_cameras) - source_count
        if novel_count == 0:
            for view in all_cameras:
                view["is_target"] = False
            return all_cameras, []

        if self.video_aug and "is_video" in all_cameras[0] and all_cameras[0]["is_video"]:
            source_indices, novel_indices = self.select_views_with_bounded_video(all_cameras, source_count)
        else:
            source_indices, novel_indices = self.select_views_with_overlap_optimization(
                all_cameras, source_count
            )
        if source_indices is None or novel_indices is None:
            for i, view in enumerate(all_cameras):
                if i < source_count:
                    view["is_target"] = False
                else:
                    view["is_target"] = True
            return all_cameras[:source_count], all_cameras[source_count:]
        context_views = [all_cameras[i] for i in source_indices]
        novel_views = [all_cameras[i] for i in novel_indices]

        assert len(context_views) + len(novel_views) == len(all_cameras)

        for view in context_views:
            view["is_target"] = False
        for view in novel_views:
            view["is_target"] = True

        return context_views, novel_views