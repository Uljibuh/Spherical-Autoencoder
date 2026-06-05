Spherical Autoencoder for MNIST

This repository contains a PyTorch implementation of a 3D Spherical Autoencoder trained on the MNIST dataset. Unlike standard autoencoders that rely on an unbounded Cartesian latent space, this model constrains the learned representations strictly to the surface of a 3D unit sphere ($S^2$ manifold).

This constraint provides a highly interpretable, bounded latent space that prevents origin collapse and is perfectly suited for 3D visualization and clustering analysis.

Overview

In traditional autoencoders, the network can minimize loss by simply pushing different classes infinitely far apart in the latent space, leading to "dead zones" and inefficient use of the representational area.

By forcing the latent space into a spherical geometry, the model is deprived of infinite distance. It must instead learn to efficiently pack and organize class clusters (digits 0-9) onto a finite, continuous surface, relying inherently on Cosine Similarity (angles) rather than Euclidean distance (straight lines).

The Core Operation: $L_2$ Normalization

The transition from a standard autoencoder to a spherical autoencoder requires no custom layers or complex loss functions. It relies entirely on a single geometric operation applied at the very end of the encoder: $L_2$ Normalization.

Given a raw, unconstrained hidden vector $h$ output by the final linear layer of the encoder, we project it onto the unit sphere to create our final latent representation $z$:

$$z = \frac{h}{||h||_2}$$

Mathematically, we calculate the $L_2$ norm (the Euclidean length of the vector from the origin) and divide the vector by its own length. This guarantees that every point passed to the decoder has a magnitude of exactly 1, forcing the entire dataset to live on the surface of a hollow sphere.

Implementation in PyTorchThis operation is executed seamlessly using torch.nn.functional.normalize:



Python

```
import torch.nn.functional as F

def encode(self, x):
    # 1. Pass data through standard linear layers
    h = self.encoder_layers(x) 
    
    # 2. Project the raw 3D output onto the surface of a unit sphere
    z = F.normalize(h, p=2, dim=1) 
    
    return z
```

```p=2``` specifies the $L_2$ norm (Euclidean length).

```dim=1``` ensures we normalize across the feature dimension (the $x, y, z$ coordinates) for each individual image in the batch, rather than normalizing across the batch dimension.



3D Visualization

Here is the learned $S^2$ manifold latent space, demonstrating how the $L_2$ normalization constrains the digit clusters to a spherical geometry:

<img width="800" height="800" alt="black_spherical_latent_space (1)" src="https://github.com/user-attachments/assets/19291d18-4f98-449e-a3eb-19962ee64360" />

