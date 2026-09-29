# Prepare repos
git clone git@github.com:syscop/nomicc26_exercises.git
git clone --branch=v1.0.0-rc --recursive git@github.com:nosnoc/nosnoc_py.git

# Download libMad release
curl -L https://github.com/madsuite-org/libMad/releases/download/v0.0.12-casadi/libMad-apple-aarch64-v0.0.12-casadi.tar.gz --output libMad.tar.gz
tar -xzf libMad.tar.gz

# Prepare environment
cd nomicc26_exercises
# Optionally use virtualenv
virtualenv -p ">=3.12" venv
source venv/bin/activate

# Install required python packages
pip install casadi caminopy
pip install -e ../nosnoc_py

# Add DYLIB_LIBRARY_PATH  environment variable to activate script
# Here you can add gurobi/LibHSL environment variables as well.
echo "" >> venv/bin/activate
echo "export DYLD_LIBRARY_PATH=\"\$DYLD_LIBRARY_PATH:$(pwd)/../libMad-apple-aarch64-v0.0.12-casadi/lib\"" >> venv/bin/activate

# Re-source the environment to set up environment variables
source venv/bin/activate

# Test your install environment
./test_install.py --ccopt --camino --nosnoc
# Optionally also test gurobi and libHSL install
# ./test_install.py --ccopt --camino --nosnoc --gurobi --libhsl
