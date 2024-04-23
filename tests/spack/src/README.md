This directory contains three files - `modules.yaml`, `spack.yaml`, and `spack.lock`. There is also a tarball, which contains the reduced software stack used as an example to showcase how these tests work.

`modules.yaml` defines the module system used (e.g. `tcl` vs `lmod`) and controls module generation. The role of this file in these tests is to match a package spec with a full module path using the defined projections in this file.

`spack.yaml` defines the specs for the 4 packages included as examples in this repo. This file is used in the `concretise_check` test to compare the abstract specs pre-concretisation with the concretised specs post-concretisation.

`spack.lock` defines the concrete specs related to the abstract specs defined in the `spack.yaml` file. There are the 4 root specs for the 4 primary packages as well as the concretised specs for all the package dependencies.

The tarball should be extracted with the following command

```
tests/spack/src > tar -xzvf . $TARBALL
```

Upon extraction there will be two areas of interest:

```
# Module files
sw_stack/modules/{architecture}/{compiler-name}/{compiler-version}/{module-category}/{module-name}/{module-version}.lua

# Software and libraries
sw_stack/software/{operating-system}-{architecture}/{compiler-name-version}/{package-name-version-hash}/
```

The first is the format for the full path of module files. This is used within the tests to match a defined package spec to a corresponding module path. On other systems this will likely be different and so the test logic will need to be changed accordingly.

The second is the directory where the binaries, libraries, etc. of a package are stored. For instance, the `bin`, `lib`, `lib64`, `include` directories would all be located in that folder. Again, the exact format of this path will be system dependent and so test logic will need to be changed accordingly.