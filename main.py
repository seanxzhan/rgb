"""
R: original image
K: stroke density
E: coarse lighting effect
S: refined lighting
O: ambient intensity, default = 0.55
I: rendered result

- S = gamma * E • K + O
- I = R • S
"""
