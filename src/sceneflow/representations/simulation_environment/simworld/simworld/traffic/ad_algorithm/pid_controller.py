"""PID controller implementation for traffic simulation.

This module provides a PID controller class that implements proportional, integral,
and derivative control to adjust vehicle behavior based on error signals.
"""
import numpy as np


class PIDController:
    """PID controller implementation for traffic simulation.

    This class implements a PID controller that adjusts vehicle behavior based on
    error signals. It provides methods to update the controller state and compute
    the control output.
    """
    def __init__(self, k_p: float, k_i: float, k_d: float):
        """Initialize the PID controller.

        Args:
            k_p: Proportional gain.
            k_i: Integral gain.
            k_d: Derivative gain.
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0

    def update(self, error: float, dt: float):
        """Update the PID controller state.

        This method updates the proportional, integral, and derivative errors
        based on the current error and time step.

        Args:
            error: The current error signal.
            dt: The time step for the update.

        Returns:
            The computed control output.
        """
        # self.i_error += error * dt
        self.i_error += error
        self.i_error = np.clip(self.i_error, -0.1, 0.1)
        # self.d_error = (error - self.p_error) / dt if dt > 0 else 0
        self.d_error = (error - self.p_error)
        self.p_error = error

        return self.k_p * self.p_error + self.k_i * self.i_error + self.k_d * self.d_error

    def reset(self):
        """Reset the PID controller state.

        This method resets the proportional, integral, and derivative errors
        to zero.
        """
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0
