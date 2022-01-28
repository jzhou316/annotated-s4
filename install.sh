#!/bin/bash

set -e

# install jax with cuda version < 11 -- https://github.com/google/jax/discussions/6236
pip install jax==0.2.21 jaxlib==0.1.71+cuda102 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# NOTE jax version >=0.2.21 is required for flax, but higher version of jax (e.g. the latest) would not work with jaxlib-v0.1.71 with cuda102
 
pip install flax==0.3.6
# NOTE later version of flax won't work, due to the `jax.config.jax_default_prng_impl` non-existence error
#      (this flag is not in jaxlib-v0.1.71, only added from jaxlib-v0.7.2, but then there is no compatible version with cuda102)

