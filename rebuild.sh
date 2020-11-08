cd build;
make -j4;
cd ../python; python3 setup.py install --user;
cd ..;
export TVM_HOME=/home/hhliao/incubator-tvm;
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH};
