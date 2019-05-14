Active GP Learning
==================================

This package provides a set of tools for performing active-learning with Gaussian Processes.

Usage
---------------

This is a work in progress but you can check the two demos in the `demo`
folder. Any feedback would be greatly appreciated.

Dependencies
---------------

You must have the following softwares in your MATLAB path:

  * minFunc: https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
  * gpml 3.6: http://www.gaussianprocess.org/gpml/code/matlab/doc
  * gpml_extensions: https://github.com/rmgarnett/gpml_extensions.git

You can run `get_dependencies.sh` to download these softwares.
But please check the links first. 

Then, running the following script: 

 - `automated_model_selection_startup.m` 

will take care of adding them to your MATLAB path.


Tests
---------------

Running `automated_model_selection_tests.m` 
will trigger a couple of tests.