
# Variational Autoencoder (VAE)

A Variational Autoencoder is a generative deep learning model that learns to encode data into a structured probabilistic latent space and decode samples from that space back into realistic outputs. It was introduced by Kingma and Welling in 2013 and remains one of the foundational approaches to generative modeling.

---

## How it Works

A standard autoencoder compresses input data into a fixed latent vector and reconstructs it. The problem is the latent space has no structure — you can't sample from it to generate new data.

A VAE fixes this by encoding input not as a single point but as a probability distribution — specifically a mean `μ` and a log variance `logvar`. A latent vector `z` is then sampled from this distribution and passed to the decoder.

```
Input x
   ↓
Encoder → μ, logvar
   ↓
z = μ + σ * ε        (reparameterization trick)
   ↓
Decoder → x̂
```

---

## The Reparameterization Trick

Sampling is not differentiable, so gradients can't flow through it during backpropagation. The trick rewrites the sample as:

```
z = μ + σ * ε,   where ε ~ N(0, 1)
```

Now `μ` and `σ` are learnable parameters gradients can flow through, and `ε` is just external noise injected at runtime.

---

## Loss Function

The VAE minimizes the **ELBO (Evidence Lower Bound)**:

```
Loss = Reconstruction Loss + KL Divergence
```

### Reconstruction Loss
Measures how well the decoder recreates the original input. Common choices are Binary Cross Entropy for images normalized to `[0, 1]` or MSE for continuous data.

### KL Divergence
Regularizes the latent space by keeping the learned distribution close to a standard normal N(0, 1):

```
KL = -0.5 * Σ (1 + logvar - μ² - exp(logvar))
```

Without this term the encoder would learn to ignore the probabilistic structure and collapse into a regular autoencoder.

---

## β-VAE

A common extension is to weight the KL term with a scalar `β`:

```
Loss = Reconstruction Loss + β * KL Divergence
```

- **β < 1** — sharper reconstructions, less organized latent space
- **β = 1** — standard VAE
- **β > 1** — more disentangled latent space, blurrier reconstructions

Higher `β` encourages the model to learn independent, interpretable factors in the latent space (e.g. separate dimensions for hair color, pose, lighting).

---

## Latent Space

Because the KL term pushes all encodings toward N(0, 1), the latent space ends up continuous and smooth. This means:

- You can **sample** a random `z ~ N(0, 1)` and decode it into a realistic output
- You can **interpolate** between two points in latent space and get smooth transitions
- You can **manipulate** specific dimensions to control attributes of generated outputs

---

## VAE vs Other Generative Models

| | VAE | GAN | Diffusion |
|---|---|---|---|
| Training stability | Stable | Unstable | Stable |
| Output sharpness | Blurry | Sharp | Very sharp |
| Latent space | Structured | Unstructured | None |
| Speed | Fast | Fast | Slow |
| Complexity | Low | Medium | High |

VAEs are blurry because the model averages over many possible reconstructions consistent with a sampled `z`. GANs solve this with a discriminator but are much harder to train. Diffusion models produce the sharpest results but are slow at inference.

---

## Common Applications

- Image generation and reconstruction
- Anomaly detection (high reconstruction loss = anomaly)
- Data augmentation
- Representation learning
- Drug and molecule discovery
- Image editing via latent space manipulation

---

## Architecture Variants

**Linear VAE** — encoder and decoder are fully connected layers. Simple but struggles with images since it treats each pixel independently with no spatial understanding.

**Convolutional VAE** — encoder uses `Conv2d` layers to downsample, decoder uses `ConvTranspose2d` layers to upsample. Much better for image data since convolutions understand local spatial structure.

**Hierarchical VAE** — uses multiple levels of latent variables to capture structure at different scales. Produces sharper results closer to diffusion model quality.

**VAE-GAN** — combines a VAE with a GAN discriminator to sharpen outputs while keeping the structured latent space. 
