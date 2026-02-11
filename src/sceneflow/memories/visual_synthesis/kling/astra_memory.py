import torch
from ...base_memory import BaseMemory


class AstraMemory(BaseMemory):
    def __init__(self, capacity=49, **kwargs):
        self.capacity = capacity
        self.history_latents = None # [C, T, H, W]

    def record(self, new_latents_squeezed, **kwargs):
        """
        Ingest new latents and update sliding window.
        new_latents_squeezed: [C, T_new, H, W]
        """
        if self.history_latents is None:
            self.history_latents = new_latents_squeezed
        else:
            self.history_latents = torch.cat([self.history_latents, new_latents_squeezed], dim=1)
        
        self.manage()

    def manage(self, **kwargs):
        """Sliding window logic: keep first frame + recent frames"""
        if self.history_latents is not None and self.history_latents.shape[1] > self.capacity:
            first_frame = self.history_latents[:, 0:1, :, :]
            recent_frames = self.history_latents[:, -(self.capacity-1):, :, :]
            self.history_latents = torch.cat([first_frame, recent_frames], dim=1)
            print(f"⚠️ History window full, keeping first frame + latest {self.capacity-1} frames")

    def select(self, target_frames_to_generate, camera_embedding_full, start_frame, modality_type):
        """
        Retrieves the prepared inputs (FramePack) for the model.
        """
        return self.prepare_framepack_sliding_window_with_camera_moe(
            self.history_latents,
            target_frames_to_generate,
            camera_embedding_full,
            start_frame,
            modality_type,
            self.capacity
        )

    # =========================================================================
    # FramePack preparation with sliding window - Mixture of Experts (MoE) version
    # =========================================================================
    
    def prepare_framepack_sliding_window_with_camera_moe(
        self,
        history_latents, 
        target_frames_to_generate, 
        camera_embedding_full, 
        start_frame, 
        modality_type, 
        max_history_frames=49):
        """FramePack sliding window mechanism - MoE version"""
        # history_latents: [C, T, H, W] current history latents
        C, T, H, W = history_latents.shape
        
        total_indices_length = 1 + 16 + 2 + 1 + target_frames_to_generate
        indices = torch.arange(0, total_indices_length)
        split_sizes = [1, 16, 2, 1, target_frames_to_generate]
        clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = \
            indices.split(split_sizes, dim=0)
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=0)
        
        if camera_embedding_full.shape[0] < total_indices_length:
            shortage = total_indices_length - camera_embedding_full.shape[0]
            padding = torch.zeros(shortage, camera_embedding_full.shape[1], 
                                dtype=camera_embedding_full.dtype, device=camera_embedding_full.device)
            camera_embedding_full = torch.cat([camera_embedding_full, padding], dim=0)
        
        combined_camera = torch.zeros(
            total_indices_length, 
            camera_embedding_full.shape[1],
            dtype=camera_embedding_full.dtype,
            device=camera_embedding_full.device)
        
        history_slice = camera_embedding_full[max(T - 19, 0):T, :].clone()
        combined_camera[19 - history_slice.shape[0]:19, :] = history_slice
        
        target_slice = camera_embedding_full[T:T + target_frames_to_generate, :].clone()
        combined_camera[19:19 + target_slice.shape[0], :] = target_slice
        
        combined_camera[:, -1] = 0.0 
        
        if T > 0:
            available_frames = min(T, 19)
            start_pos = 19 - available_frames
            combined_camera[start_pos:19, -1] = 1.0 
        
        clean_latents_combined = torch.zeros(C, 19, H, W, dtype=history_latents.dtype, device=history_latents.device)
        
        if T > 0:
            available_frames = min(T, 19)
            start_pos = 19 - available_frames
            clean_latents_combined[:, start_pos:, :, :] = history_latents[:, -available_frames:, :, :]
        
        clean_latents_4x = clean_latents_combined[:, 0:16, :, :]
        clean_latents_2x = clean_latents_combined[:, 16:18, :, :]
        clean_latents_1x = clean_latents_combined[:, 18:19, :, :]
        
        if T > 0:
            start_latent = history_latents[:, 0:1, :, :]
        else:
            start_latent = torch.zeros(C, 1, H, W, dtype=history_latents.dtype, device=history_latents.device)
        
        clean_latents = torch.cat([start_latent, clean_latents_1x], dim=1)
        
        return {
            'latent_indices': latent_indices,
            'clean_latents': clean_latents,
            'clean_latents_2x': clean_latents_2x,
            'clean_latents_4x': clean_latents_4x,
            'clean_latent_indices': clean_latent_indices,
            'clean_latent_2x_indices': clean_latent_2x_indices,
            'clean_latent_4x_indices': clean_latent_4x_indices,
            'camera_embedding': combined_camera,
            'modality_type': modality_type,
            'current_length': T,
            'next_length': T + target_frames_to_generate
        }
