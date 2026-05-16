"""
    Tarang.TransformOps

Facade for Tarang transform operations.

`Tarang.Transforms` is already a binding imported from PencilFFTs, so this
namespace uses `TransformOps` to avoid shadowing third-party internals.
"""
module TransformOps
import ..Tarang:
    forward_transform!, backward_transform!,
    distributed_forward_transform!, distributed_backward_transform!,
    setup_pencil_fft_transforms_2d!, setup_pencil_fft_transforms_3d!,
    RealFourier, ComplexFourier, Fourier,
    ChebyshevT, ChebyshevU, ChebyshevV, Legendre, Jacobi

export
    forward_transform!, backward_transform!,
    distributed_forward_transform!, distributed_backward_transform!,
    setup_pencil_fft_transforms_2d!, setup_pencil_fft_transforms_3d!,
    RealFourier, ComplexFourier, Fourier,
    ChebyshevT, ChebyshevU, ChebyshevV, Legendre, Jacobi
end
