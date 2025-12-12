import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

def gauss_newton_decomposition(low_img, high_img, mu_basis_low, mu_basis_high, 
                                max_iter=10, tol=1e-6):
    """
    Performs Gauss-Newton material decomposition on 2D image data.

    Parameters:
    - low_img: 2D numpy array (low energy image)
    - high_img: 2D numpy array (high energy image)
    - mu_basis_low: [mu_PMMA_low, mu_Al_low]
    - mu_basis_high: [mu_PMMA_high, mu_Al_high]
    """
    img_shape = low_img.shape
    # Initialize basis weights: w1 for PMMA, w2 for Aluminum
    w = np.ones((*img_shape, 2)) * 0.5

    # Convert basis to matrix form
    B = np.array([mu_basis_low, mu_basis_high])  # shape: (2 energies, 2 materials)

    for i in range(max_iter):
        # Forward model: estimate the predicted attenuation at both energies
        f = np.einsum('ij,xyj->xyi', B, w)

        # Residual: difference between measured and predicted attenuation
        r = np.stack([low_img, high_img], axis=-1) - f  # shape: (x, y, 2)

        # Compute Jacobian (same for every voxel)
        J = B  # shape: (2, 2)

        # Compute Gauss-Newton update
        JTJ_inv = np.linalg.inv(J.T @ J)  # 2x2
        update = np.einsum('ij,jk,xyk->xyi', JTJ_inv, J.T, r)  # shape: (x, y, 2)

        # Update weights
        w += update

        # Check convergence
        if np.linalg.norm(update) < tol:
            print(f"Converged in {i+1} iterations.")
            break

    return w[..., 0], w[..., 1]  # PMMA and Aluminum maps

def direct_material_decomposition(low_img, high_img, mu_basis_low, mu_basis_high):
    """
    Direct linear least squares material decomposition.
    Solves w = (Bᵗ B)⁻¹ Bᵗ μ for each pixel.
    """
    # Stack input images into a (x, y, 2) array
    mu = np.stack([low_img, high_img], axis=-1)  # Shape: (x, y, 2)

    # Basis matrix B (shape 2x2)
    B = np.array([mu_basis_low, mu_basis_high])  # [[mu_PMMA_low, mu_PTFE_low], [mu_PMMA_high, mu_PTFE_high]]

    # Precompute the pseudo-inverse of B
    B_pinv = np.linalg.inv(B.T @ B) @ B.T    # Shape: (2, 2)

    # Apply the pseudo-inverse to every pixel
    w = np.einsum('ij,xyj->xyi', B_pinv, mu)  # Shape: (x, y, 2)

    return w[..., 0], w[..., 1]  # PMMA map, PTFE map



# Load or create your input image data (2D arrays)

low_image = imageio.imread(".\phantom_avg24.tif")
high_image = imageio.imread(".\phantom_avg38.tif")

# Define known basis material values at both energies
mu_PMMA_low = 0.49028
mu_PMMA_high = 0.28977
mu_PTFE_low = 1.34734
mu_PTFE_high = 0.60476

# Run decomposition
w_PMMA, w_PTFE = gauss_newton_decomposition(
    low_image,
    high_image,
    mu_basis_low=[mu_PMMA_low, mu_PTFE_low],
    mu_basis_high=[mu_PMMA_high, mu_PTFE_high]
)

# Plot result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(w_PMMA)
plt.title("PMMA Map")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(w_PTFE)
plt.title("PTFE Map")
plt.colorbar()
plt.tight_layout()
plt.show()

# Energies where you would like to see your VMIs
Es = np.arange(20,51,2)
mu_pe = []
mu_pom = []
mu_nylon = []
mu_ptfe = []
mu_pmma = []
mu_water = []
with open(".\square_phantom_materials.txt") as f:
    for line in f:
        entries = line.split(',')
        E = int(float(entries[0].strip()))

        if E in Es:
            # print(E)
            mu_pe.append(float(entries[1])* 0.93)
            mu_pom.append(float(entries[2]) * 1.42)    #densities
            mu_nylon.append(float(entries[3]) * 1.14)
            mu_ptfe.append(float(entries[4]) * 2.14)
            mu_pmma.append(float(entries[5]) * 1.19)
            mu_water.append(float(entries[6]))

            
for i in range(len(Es)):
    img = Image.fromarray(w_PMMA*mu_pmma[i]+w_PTFE*mu_ptfe[i])
    img.save('./VMI/mono_'+str(Es[i])+'keV.tif')