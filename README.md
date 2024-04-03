# Diffusion Maps
Non-linear dimension reduction using diffusion maps. We can them embed high dimension data onto a manifold in a lower dimension. This works even when there is a lot of noise petrubing the manifold.

# Example
We construct a swiss_roll data set and use a diffusion map to "unroll" it and embed it into R^2. 
```py
swiss_roll, _ = make_swiss_roll(n_samples=1500, noise=noise)
d_map = diffusion_map(swiss_roll, 2, t=1, alpha=0.9)
```
Here `t` is the number of steps taken in the diffusion map and `alpha` is the normalization parameter for gaussian kernel function used.

![alt text](/example.png)
