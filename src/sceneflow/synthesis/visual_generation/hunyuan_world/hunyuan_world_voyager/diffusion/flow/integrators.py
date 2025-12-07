import torch as th


class sde:
    """Stochastic Differential Equation (SDE) solver class
    
    This class implements numerical solvers for stochastic differential equations
    used in diffusion models and flow matching. It supports different sampling
    methods like Euler-Maruyama and Heun schemes.
    """

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        """Initialize SDE solver with drift and diffusion functions
        
        Args:
            drift: Function that computes the drift term (deterministic part)
            diffusion: Function that computes the diffusion term (stochastic part)
            t0 (float): Initial time point
            t1 (float): Final time point
            num_steps (int): Number of integration steps
            sampler_type (str): Type of numerical scheme ("Euler" or "Heun")
        """
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        # Create time grid for integration
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]  # Time step size
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        """Euler-Maruyama scheme for SDE integration
        
        This is a first-order numerical scheme for solving SDEs.
        It approximates the solution using a predictor-corrector approach.
        
        Args:
            x (torch.Tensor): Current state
            mean_x (torch.Tensor): Mean prediction (used for predictor-corrector)
            t (float): Current time point
            model: Neural network model for drift computation
            **model_kwargs: Additional arguments for the model
            
        Returns:
            tuple: (updated_state, updated_mean)
        """
        # Generate random noise for stochastic term
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        
        # Compute Wiener increment
        dw = w_cur * th.sqrt(self.dt)
        
        # Compute drift and diffusion terms
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        
        # Predictor step: compute mean prediction
        mean_x = x + drift * self.dt
        
        # Corrector step: add stochastic term
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        """Heun scheme for SDE integration (second-order accurate)
        
        This is a second-order numerical scheme that provides better accuracy
        than Euler-Maruyama by using a two-stage predictor-corrector approach.
        
        Args:
            x (torch.Tensor): Current state
            _: Unused parameter (for compatibility)
            t (float): Current time point
            model: Neural network model for drift computation
            **model_kwargs: Additional arguments for the model
            
        Returns:
            tuple: (updated_state, intermediate_state)
        """
        # Generate random noise
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        
        # Compute diffusion term
        diffusion = self.diffusion(x, t_cur)
        
        # First stage: add noise to get intermediate state
        xhat = x + th.sqrt(2 * diffusion) * dw
        
        # Second stage: compute drift at intermediate state
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        
        # Third stage: compute drift at predicted state
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        
        # Final update: average the two drift estimates
        return (
            xhat + 0.5 * self.dt * (K1 + K2),
            xhat,
        )  # at last time point we do not perform the heun step

    def __forward_fn(self):
        """Get the appropriate sampling function based on sampler type
        
        Returns:
            function: The selected numerical integration scheme
            
        Raises:
            NotImplementedError: If the sampler type is not supported
        """
        # TODO: generalize here by adding all private functions ending with steps to it
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Sampler type not implemented.")

        return sampler

    def sample(self, init, model, **model_kwargs):
        """Forward integration loop for SDE
        
        This method performs the complete forward integration of the SDE
        from initial condition to final time.
        
        Args:
            init (torch.Tensor): Initial condition
            model: Neural network model for drift computation
            **model_kwargs: Additional arguments for the model
            
        Returns:
            list: List of states at each time step
        """
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        
        # Integrate over all time steps except the last one
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples


class ode:
    """Ordinary Differential Equation (ODE) solver class
    
    This class implements numerical solvers for ordinary differential equations
    used in deterministic flow matching and probability flow ODEs.
    """

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_shifting_factor=None,
    ):
        """Initialize ODE solver with drift function
        
        Args:
            drift: Function that computes the drift term
            t0 (float): Initial time point
            t1 (float): Final time point
            sampler_type (str): Type of ODE solver method
            num_steps (int): Number of integration steps
            atol (float): Absolute tolerance for adaptive step sizing
            rtol (float): Relative tolerance for adaptive step sizing
            time_shifting_factor (float, optional): Factor for time rescaling
        """
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        # Create time grid
        self.t = th.linspace(t0, t1, num_steps)
        
        # Apply time shifting if specified (for better numerical stability)
        if time_shifting_factor:
            self.t = self.t / (self.t + time_shifting_factor -
                               time_shifting_factor * self.t)
        
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        """Solve ODE using torchdiffeq library
        
        This method uses the torchdiffeq library to solve the ODE with
        adaptive step sizing and high-order numerical schemes.
        
        Args:
            x (torch.Tensor or tuple): Initial condition(s)
            model: Neural network model for drift computation
            **model_kwargs: Additional arguments for the model
            
        Returns:
            torch.Tensor: Solution trajectory at all time points
        """
        from torchdiffeq import odeint
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            """ODE function for torchdiffeq
            
            This function computes the right-hand side of the ODE at time t.
            """
            # Expand time to match batch dimension
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x,
                                                                   tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        # Set tolerances for adaptive step sizing
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        
        # Solve ODE using torchdiffeq
        samples = odeint(_fn, x, t, method=self.sampler_type,
                         atol=atol, rtol=rtol)
        return samples

    def sample_with_step_fn(self, x, step_fn):
        """Solve ODE using a custom step function
        
        This method allows using a custom step function instead of the
        default drift function, providing more flexibility.
        
        Args:
            x (torch.Tensor or tuple): Initial condition(s)
            step_fn: Custom function that computes the ODE right-hand side
            
        Returns:
            torch.Tensor: Solution trajectory at all time points
        """
        from torchdiffeq import odeint
        device = x[0].device if isinstance(x, tuple) else x.device
        t = self.t.to(device)
        
        # Set tolerances for adaptive step sizing
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        
        # Solve ODE using custom step function
        samples = odeint(
            step_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples
