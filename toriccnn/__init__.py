"""ToricCNN: convolutional networks on sphere-type surfaces via toric covers.

Pure-Python/PyTorch port of Maron et al., "Convolutional Neural Networks on
Surfaces via Seamless Toric Covers" (SIGGRAPH 2017). Pipeline:
``toriccnn-make-data`` -> ``toriccnn-train`` -> ``toriccnn-predict`` ->
``toriccnn-aggregate``.
"""

__version__ = "0.1.0"
