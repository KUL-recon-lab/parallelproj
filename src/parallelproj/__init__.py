from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("parallelproj")
except PackageNotFoundError:
    __version__ = "unknown"

from .backend import (
    to_numpy_array,
    empty_cuda_cache,
    Array,
)

from .operators import (
    LinearOperator,
    MatrixOperator,
    ElementwiseMultiplicationOperator,
    TOFNonTOFElementwiseMultiplicationOperator,
    GaussianFilterOperator,
    CompositeLinearOperator,
    VstackOperator,
    LinearOperatorSequence,
    FiniteForwardDifference,
    GradientFieldProjectionOperator,
)

from .projectors import (
    ParallelViewProjector2D,
    ParallelViewProjector3D,
    RegularPolygonPETProjector,
    ListmodePETProjector,
    EqualBlockPETProjector,
)

from .pet_scanners import (
    RegularPolygonPETScannerModule,
    RegularPolygonPETScannerGeometry,
    DemoPETScannerGeometry,
    BlockPETScannerModule,
    ModularizedPETScannerGeometry,
)
from .pet_lors import (
    SinogramSpatialAxisOrder,
    RegularPolygonPETLORDescriptor,
    EqualBlockPETLORDescriptor,
)

from .tof import TOFParameters

__all__ = [
    "to_numpy_array",
    "empty_cuda_cache",
    "LinearOperator",
    "MatrixOperator",
    "ElementwiseMultiplicationOperator",
    "TOFNonTOFElementwiseMultiplicationOperator",
    "GaussianFilterOperator",
    "CompositeLinearOperator",
    "VstackOperator",
    "LinearOperatorSequence",
    "FiniteForwardDifference",
    "GradientFieldProjectionOperator",
    "ParallelViewProjector2D",
    "ParallelViewProjector3D",
    "RegularPolygonPETProjector",
    "EqualBlockPETProjector",
    "ListmodePETProjector",
    "RegularPolygonPETScannerModule",
    "RegularPolygonPETScannerGeometry",
    "DemoPETScannerGeometry",
    "BlockPETScannerModule",
    "ModularizedPETScannerGeometry",
    "TOFParameters",
    "SinogramSpatialAxisOrder",
    "RegularPolygonPETLORDescriptor",
    "EqualBlockPETLORDescriptor",
    "Array",
]
