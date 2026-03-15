import math
import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur
from scipy.ndimage import median_filter
from visualizations import Transformations, Visualization

class LuptonRgbTransform:
    def __init__(self, stretch=0.5, Q=10):
        self.stretch = stretch
        self.Q = Q
        
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            # Transformations.channels_to_rgb expects numpy arrays.
            image_np = image.detach().cpu().numpy()
        else:
            image_np = np.asarray(image)

        rgb_image = Transformations.channels_to_rgb(image_np, stretch=self.stretch, Q=self.Q)
        #enforce CHW and float32 output
        rgb_image = np.transpose(rgb_image, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(rgb_image)

class MultiScaleUnsharpMaskTransform:
    def __init__(
        self,
        sigmas=(2.5, 5.5),
        amounts=(1.0, 2.0),
        thresholds=(0.005, 0.002),
        clip_percentiles=(0.1, 99.9),
        z_amount_boost=0.0,
    ):
        self.sigmas = sigmas
        self.amounts = torch.tensor(amounts)
        self.thresholds = torch.tensor(thresholds)
        self.clip_percentiles = clip_percentiles
        self.z_amount_boost = z_amount_boost

    @torch.no_grad()
    def __call__(self, image):
        device = image.device

        boosts = torch.tensor([0.0, 0.0, self.z_amount_boost], device=device).view(3, 1, 1)
        detail_sum = torch.zeros_like(image)

        for sigma, amount, thr in zip(self.sigmas, self.amounts, self.thresholds):
            k_size = int(4 * sigma + 1)
            if k_size % 2 == 0:
                k_size += 1

            blurred = gaussian_blur(image, [k_size, k_size], [sigma, sigma])
            diff = image - blurred
            mask = torch.abs(diff) > thr
            curr_amount = amount + boosts
            detail_sum += diff * mask * curr_amount

        out = image + detail_sum

        # Percentile clipping via kthvalue — O(n) instead of O(n log n) quantile
        q_low = self.clip_percentiles[0] / 100.0
        q_high = self.clip_percentiles[1] / 100.0

        flat_out = out.view(3, -1)
        n = flat_out.shape[1]
        k_low = max(1, int(q_low * n))
        k_high = min(n, int(q_high * n))
        lo = torch.kthvalue(flat_out, k_low, dim=1, keepdim=True).values.view(3, 1, 1)
        hi = torch.kthvalue(flat_out, k_high, dim=1, keepdim=True).values.view(3, 1, 1)

        return torch.clamp(out, lo, hi)


class SkySubstractTransform:
    @torch.no_grad()
    def __call__(self, image):
        # image: (3, H, W) torch.Tensor
        sky = image.flatten(1).median(dim=1).values.view(-1, 1, 1)
        return image - sky

class EnsureCHWTransform:
    def __call__(self, image):
        return image.permute(2, 0, 1) if isinstance(image, torch.Tensor) and image.ndim == 3 and image.shape[-1] == 3 else image

class ScaleToUnitIntervalTransform:
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:  # Avoid division by zero
                return (image - min_val) / (max_val - min_val)
            else:
                return image  # If all values are the same, return the original image
        else:
            return image
        
class LinearConstantStretchTransform:
    def __init__(self, b_th=0.0, w_th=255.0):
        self.b_th = b_th
        self.w_th = w_th

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            if self.w_th > self.b_th:  # Avoid division by zero
                # scaled = (image - self.low) * (255.0 / (self.high - self.low))
                scaled = self._contrast_stretch(image, self.b_th, self.w_th)
                return scaled
            else:
                return image  # If all values are the same, return the original image
        else:
            return image

    def _contrast_stretch(self, image: torch.Tensor, b_th: float, w_th: float) -> torch.Tensor:
        scaled = (image.float() - b_th) * (255.0 / (w_th - b_th))   # linear rescaling
        return scaled                

# ...existing code...

class MedianFilterTransform:
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    @torch.no_grad()
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            device = image.device
            dtype = image.dtype
            # Convert to numpy for scipy, apply filter per channel (CHW layout)
            image_np = image.detach().cpu().numpy()
            filtered = np.stack([
                median_filter(image_np[c], size=self.kernel_size, mode='reflect')
                for c in range(image_np.shape[0])
            ])
            return torch.from_numpy(filtered).to(device=device, dtype=dtype)
        else:
            return median_filter(image, size=[self.kernel_size, self.kernel_size], mode='reflect')
        
class ClipPercentilesTransform:
    def __init__(self, clip_percentiles=(0.1, 99.9)):
        self.clip_percentiles = clip_percentiles
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            device = image.device
            dtype = image.dtype
            flat_image = image.view(image.shape[0], -1)
            n = flat_image.shape[1]
            q_low = self.clip_percentiles[0] / 100.0
            q_high = self.clip_percentiles[1] / 100.0
            k_low = max(1, int(q_low * n))
            k_high = min(n, int(q_high * n))
            lo = torch.kthvalue(flat_image, k_low, dim=1, keepdim=True).values.view(-1, 1, 1)
            hi = torch.kthvalue(flat_image, k_high, dim=1, keepdim=True).values.view(-1, 1, 1)
            return torch.clamp(image, lo, hi)
        else:
            # For PIL images or numpy arrays, we can use numpy's percentile function for simplicity
            low = np.percentile(image, self.clip_percentiles[0])
            high = np.percentile(image, self.clip_percentiles[1])
            return np.clip(image, low, high)

class MaskOutsideRadialProfileTransform:
    def __init__(self, threshold=None):
        self.threshold = threshold

    def __call__(self, image):
        radial_profiles = []
        for channel in range(image.shape[0]):
            radial_profile = Visualization.calculate_radial_profile(image[channel])
            radial_profiles.append(radial_profile)

        #Based on the radial profiles, draw a circle at the radius where the profile drops below 5% of its maximum value for each channel. Plot the results and print the radius values for each channel.
        radii=[]
        for channel, radial_profile in enumerate(radial_profiles):
            max_value = np.max(radial_profile)
            threshold = self.threshold * max_value
            indices = np.where(radial_profile < threshold)[0]
            radius = indices[0] if len(indices) > 0 else len(radial_profile) - 1
            radii.append(radius)

        self.max_radius = max(radii)
        # Create a mask that is True for pixels within the max_radius and False for pixels outside
        H, W = image.shape[1], image.shape[2]
        y, x = torch.meshgrid(torch.arange(H, device=image.device), torch.arange(W, device=image.device))
        center_y, center_x = H // 2, W // 2
        distance_from_center = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask = distance_from_center <= self.max_radius
        # Apply the mask to each channel
        masked_image = image * mask.unsqueeze(0)
        return masked_image
            

