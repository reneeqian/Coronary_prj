# This script checks that the medical imaging environment is set up correctly.
# It imports key libraries and prints their versions.
import numpy as np
import torch
import monai
import nibabel
import SimpleITK

print("✅ Medical imaging environment ready")
print("Python:", __import__("sys").version)
print("Torch:", torch.__version__)
print("MONAI:", monai.__version__)
print("NumPy:", np.__version__)
print("Nibabel:", nibabel.__version__)
print("SimpleITK:", SimpleITK.__version__)

