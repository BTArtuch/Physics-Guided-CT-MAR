# Physics-Guided Metal Artifact Reduction (MAR)

Metal implants, such as hip prostheses, dental fillings, and surgical clips, introduce severe corruption into Computed Tomography (CT) images, manifesting primarily as streaking, dark bands, and severe shading between high-density objects. The physical phenomena underlying these artifacts include photon starvation, where metallic objects absorb nearly the entire X-ray beam energy, and beam hardening, where lower-energy photons are preferentially absorbed as they pass through dense metal. 

Standard deep learning approaches treat Metal Artifact Reduction (MAR) purely as an image-to-image translation task. Because these models operate exclusively in the image space without physical grounding, they frequently hallucinate anatomical tissue, erase real pathological structures, or generate non-physical density variations near implant boundaries.

## Framework Overview
This project proposes a dual-domain, physics-guided reconstruction framework designed to resolve the limitations of purely data-driven models. The architecture couples a deep image-to-image restoration generator operating in the image space with a custom C++/CUDA forward-projection layer operating in the projection space, also known as the sinogram domain. By dynamically simulating X-ray line integrals over predicted volumes, the model computes a Radon Transform Penalty, enforcing rigorous physical consistency between generated attenuation maps and measured raw projection data.

## Methodology

### Deep Learning Component (Image Domain)
*   The framework utilizes a high-capacity 3D generator (e.g., 3D U-Net backbone) to restore corrupted volumetric CT scans into clean, high-fidelity anatomical volumes.
*   The network processes contiguous 3D CT volumes by accepting a metal-corrupted CT volume alongside a binary metal mask to predict a corrected clean attenuation volume.
*   Direct supervision is provided via structural loss terms combining L1 distance and MS-SSIM against uncorrupted ground-truth scans.

### Physics-Guided Loss Component (Projection Domain)
*   To prevent phantom artifacts and physical hallucinations, the network must verify that its predicted 3D density volume yields logical X-ray projections that match the original uncorrupted detector measurements.
*   The framework employs Joseph's Ray-Casting Algorithm, which intersects rays with voxel boundaries and interpolates attenuation values linearly along the primary ray axis, providing an accurate, computationally efficient approximation of X-ray line integrals.
*   A custom high-performance, GPU-accelerated forward-projection operator bridges directly into PyTorch's `torch.autograd` engine via pybind11.
*   During the forward pass, the custom layer launches a parallel grid of CUDA threads where each thread tracks a single X-ray path from the source position through the 3D volume grid to an individual detector element.
*   During the backward pass, the layer computes the analytical gradient of ray integrals with respect to every voxel intensity, backpropagating sinogram-space error gradients directly into the 3D generator weights.
*   The Radon Transform Loss acts as a penalty term calculating the Mean Squared Error (MSE) between the synthesized sinogram and the valid (uncorrupted) regions of the measured raw projection data.

## References
* **P. M. Joseph**, "An Improved Algorithm for Reprojecting Rays through Pixel Images," in *IEEE Transactions on Medical Imaging*, vol. 1, no. 3, pp. 192-196, Nov. 1982. DOI: [10.1109/TMI.1982.4307572](https://doi.org/10.1109/TMI.1982.4307572)
* **Hsieh, J.** (2009). *Computed Tomography: Principles, Design, Artifacts, and Recent Advances* (2nd ed.).
* **Kirk, D. B., & Hwu, W. M. W.** (2022). "Programming massively parallel processors: A hands-on approach" (4th ed.). Morgan Kaufmann.