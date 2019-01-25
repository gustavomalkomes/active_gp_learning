# A simple script to clone/download dependencies
# Make sure you check the original links before using this script

gpml_extensions="gpml_extensions"
minFunc="minFunc_2012"
gpml="gpml-matlab-v3.6"
mgp="mgp"

cd ..

# https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
if [ ! -d "$minFunc" ]; then
    wget https://www.cs.ubc.ca/~schmidtm/Software/minFunc_2012.zip
    unzip -a minFunc_2012.zipu
    rm minFunc_2012.zip
fi

# http://gaussianprocess.org/gpml/code/matlab/doc/index.html
if [ ! -d "$gpml" ]; then
    wget http://gaussianprocess.org/gpml/code/matlab/release/gpml-matlab-v3.6-2015-07-07.zip
    unzip -a gpml-matlab-v3.6-2015-07-07.zip
    rm gpml-matlab-v3.6-2015-07-07.zip
    mv gpml-matlab-v3.6-2015-07-07 gpml-matlab-v3.6
fi

# https://github.com/rmgarnett/mgp
if [ ! -d "$mgp" ]; then
    git clone https://github.com/rmgarnett/mgp.git
fi

# https://github.com/rmgarnett/gpml_extensions
if [ ! -d "$gpml_extensions" ]; then
    git clone https://github.com/rmgarnett/gpml_extensions.git
fi
