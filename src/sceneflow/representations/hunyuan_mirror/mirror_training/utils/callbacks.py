"""Custom callbacks for training."""

from lightning.pytorch.callbacks import TQDMProgressBar


class StepBasedProgressBar(TQDMProgressBar):
    """Progress bar that shows steps instead of epochs when max_steps is set."""
    
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
    
    def get_metrics(self, trainer, pl_module):
        """Override to customize displayed metrics."""
        items = super().get_metrics(trainer, pl_module)
        # Remove version number if present
        items.pop("v_num", None)
        return items
    
    def init_train_tqdm(self):
        """Override the train progress bar to show steps instead of batches per epoch."""
        bar = super().init_train_tqdm()
        
        # If max_steps is set, change the progress bar to show total steps
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            bar.reset(total=self.trainer.max_steps)
            bar.initial = self.trainer.global_step
            bar.n = self.trainer.global_step
            # Update the bar format to show steps
            bar.set_description(f"Step")
            bar.refresh()
        
        return bar
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Override to prevent resetting progress bar at epoch start."""
        # Only call parent if not using max_steps (to avoid epoch-based reset)
        if not (trainer.max_steps and trainer.max_steps > 0):
            super().on_train_epoch_start(trainer, pl_module)
        else:
            # Manually set the total and position for step-based progress
            if self.train_progress_bar is not None:
                self.train_progress_bar.reset(total=trainer.max_steps)
                self.train_progress_bar.initial = trainer.global_step
                self.train_progress_bar.n = trainer.global_step
                self.train_progress_bar.set_description(f"Step")
                self.train_progress_bar.refresh()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update progress bar after each training batch."""
        # Update the progress bar to current global step if using max_steps
        if trainer.max_steps and trainer.max_steps > 0:
            if self.train_progress_bar is not None:
                self.train_progress_bar.n = trainer.global_step
                self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
                self.train_progress_bar.refresh()
        else:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

