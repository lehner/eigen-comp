# eigen-comp
This repository contains a standalone QCD eigenvector compressor and decompressor.
It achieves substantial compression by combining three ideas:

* Using local coherence of low modes.  We express high eigenvectors as linear combinations
  of blocked low eigenvectors.  This alone achieves about a factor of 3-4 compression for RBC/UKQCD's 48^3 eigenvectors.
  
* Using 2-byte fixed precision with shared exponents.  The original eigenvector data is
  received in single precision, i.e., with a 8-bit exponent and 24-bit mantissa.  We save
  the bulk of data in a 16-bit mantissa format and share the exponent among multiple floats.
  This achieves an additional factor of 2 compression for RBC/UKQCD's 48^3 eigenvectors.
  
* Variable precision over the eigenspace.  We save the data corresponding to the lowest eigenvectors
  in the original single precision and use fixed precision for the bulk of the data.
  
Combining these ideas achieves a 85% reduction in size for RBC/UKQCD's 48^3 eigenvectors and 92% reduction
in size for RBC/UKQCD's 64^3 eigenvectors with acceptable precision for both CG and exact all-to-all low-mode
reconstruction.

# Plots
A brief demonstration can be found [here](https://github.com/lehner/eigen-comp/blob/master/tests/plots.pdf).

# Contributors
* Chulwoo Jung
* Christoph Lehner

