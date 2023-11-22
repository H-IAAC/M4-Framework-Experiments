export PORT=8888
docker run --gpus 'device=0' -it --rm  -e SHELL="/bin/bash" -e HOME=`pwd` -p ${PORT}:${PORT} -v`pwd`:`pwd` -w`pwd` -u $(id -u ${USER}):$(id -g ${USER}) experiment-executor python -m jupyterlab --port ${PORT} --no-browser --ip='0.0.0.0' --NotebookApp.token='' --NotebookApp.password=''
