`phase_bits.npy` contains a bit-packed phase mask. Exactly, `np.packbits(sdf >= 0)`.
It can be extracted using `np.unpackbits(np.load(...))`
