[//]: # (![Coverage Report]&#40;./assets/coverage.svg&#41;)

# <span style="color:#0E6FFF">NRN</span> DOCUMENTATION

_NRN ([Neural Reasoning Networks](https://arxiv.org/abs/2410.07966))_ is a **PyTorch** framework for developing Neuro-Symbolic AI systems
based on [Weighted Lukasiewicz Logic](https://arxiv.org/abs/2006.13155).  
The design principles of _NRN_ provide computational efficiency for neuro‑symbolic AI with GPU scaling.

### Design Principles

- _Neural == Symbolic_: Symbolic operations should not deviate computationally from Neural operations and leverage PyTorch directly
- _Masked Tensors_: Reasoning Networks use tensors and masking to represent any logical structure _and_ leverage GPU optimized computations
- _Neural -> Symbolic Extension_: Symbolic operations in _NRN_ are PyTorch Modules and can therefore integrate with existing Deep Neural Networks seamlessly

With these principles, _NRN_ is able
to extend and integrate with our current state-of-the-art technologies that leverage advances in 
Deep Learning.  Neural Reasoning Networks developed with _NRN_ can scale with
multi-GPU support.  Finally, those familiar with PyTorch development principles will have only a small step
in skill building to develop with _NRN_.

### Documentation

You’re reading the **official NRN docs**. This site includes installation, quickstart guides, tutorials, and a full API reference.

> **Status:** Alpha — APIs may change; we keep docs aligned with releases. If something looks off, please open an issue or PR.

### Tutorial

There are several tutorials demonstrating how to use the R-NRN algorithm
in multiple use cases.

[Tutorial Source](./tutorials/notebooks.md)

### Data Science

To understand the basics of Neural Reasoning Networks
check out the [Data Science](./ds/rn.md) section, which gives an introduction to some of the
models developed so far using NRN.
