from .pipelines import HunyuanVideoPipeline
from .schedulers import FlowMatchDiscreteScheduler
from .flow.transport import *


def create_transport(
        *,
        path_type,
        prediction,
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        snr_type="uniform",
        shift=1.0,
        video_shift=None,
        reverse=False,
):
    """Create a transport configuration for flow-based diffusion models
    
    This function configures the transport mechanism used in flow matching
    diffusion models, setting up prediction types, loss weights, and noise schedules.
    
    Args:
        path_type: Type of path for the flow (linear, gvp, vp)
        prediction: Prediction target (noise, score, velocity)
        loss_weight: Loss weighting strategy (velocity, likelihood, None)
        train_eps: Training epsilon value for numerical stability
        sample_eps: Sampling epsilon value for numerical stability
        snr_type: Signal-to-noise ratio type (uniform, lognorm)
        shift: General shift parameter
        video_shift: Video-specific shift parameter
        reverse: Whether to reverse the flow direction
    """
    # Determine model prediction type based on input
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    # Set loss weighting strategy
    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    # Configure signal-to-noise ratio type
    if snr_type == "lognorm":
        snr_type = SNRType.LOGNORM
    elif snr_type == "uniform":
        snr_type = SNRType.UNIFORM
    else:
        raise ValueError(f"Invalid snr type {snr_type}")

    # Use general shift if video-specific shift is not provided
    if video_shift is None:
        video_shift = shift

    # Map string path types to enum values
    path_choice = {
        "linear": PathType.LINEAR,
        "gvp": PathType.GVP,
        "vp": PathType.VP,
    }

    path_type = path_choice[path_type.lower()]

    # Set epsilon values based on path type and model type for numerical stability
    if path_type in [PathType.VP]:
        # VP path requires small epsilon values for stability
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        # GVP and LINEAR paths with non-velocity models need moderate epsilon
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    else:  # velocity & [GVP, LINEAR] is stable everywhere
        # Velocity models with GVP/LINEAR paths are stable without epsilon
        train_eps = 0
        sample_eps = 0

    # Create and return the transport state configuration
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        snr_type=snr_type,
        shift=shift,
        video_shift=video_shift,
        reverse=reverse,
    )

    return state


def load_denoiser(args):
    """Load and configure a denoiser based on command line arguments
    
    This function creates a denoiser instance based on the specified type
    and configuration parameters from the argument parser.
    
    Args:
        args: Argument parser object containing denoiser configuration
        
    Returns:
        Configured denoiser instance
        
    Raises:
        ValueError: If an unknown denoise type is specified
    """
    # Create flow-based denoiser if specified
    if args.denoise_type == "flow":
        denoiser = create_transport(path_type=args.flow_path_type,
                                    prediction=args.flow_predict_type,
                                    loss_weight=args.flow_loss_weight,
                                    train_eps=args.flow_train_eps,
                                    sample_eps=args.flow_sample_eps,
                                    snr_type=args.flow_snr_type,
                                    shift=args.flow_shift,
                                    video_shift=args.flow_shift,
                                    reverse=args.flow_reverse,
                                    )
    else:
        # Raise error for unsupported denoiser types
        raise ValueError(f"Unknown denoise type: {args.denoise_type}")
    return denoiser
