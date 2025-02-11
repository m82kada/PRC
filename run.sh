#!/bin/bash

tfds="./tensorflow_datasets:/home/user/tensorflow_datasets"
tfds="/raid5/mkada/tensorflow_datasets:/home/user/tensorflow_datasets"
# If you want to put the workdir on external storage, you can use the following line.
# workdir=" -v /workdir:/home/user/app/workdir "



if [ $# -lt 2 ]; then
	echo "Error: You must write arguments like './run.sh <gpu> <app> (<app arg>)'"
	echo "Example) ./run.sh 0 zsh"
	echo "         ./run.sh all zsh"
	echo "         ./run.sh none tensorboard"
	echo "         ./run.sh none notebook"
	exit
fi

gpu=\'\"device=0\"\'

if [ "$1" == "all" ]; then
	gpu=" --gpus all "
elif [ "$1" == "none" ]; then
	gpu=" "
else
	gpu=" --gpus '\"device=$1\"' "
fi

image=`cat docker/image.txt`

if [ "$2" == "zsh" ]; then
	eval "docker run $gpu --name vmoe --rm -it -v $(pwd)/app:/home/user/app -v $tfds $workdir $image zsh"
elif [ "$2" == "zsh2" ]; then
	eval "docker run $gpu --name vmoe2 --rm -it -v $(pwd)/app:/home/user/app -v $tfds $workdir $image zsh"
elif [ "$2" == "zsh3" ]; then
	eval "docker run $gpu --name vmoe3 --rm -it -v $(pwd)/app:/home/user/app -v $tfds $workdir $image zsh"
elif [ "$2" == "zsh4" ]; then
	eval "docker run $gpu --name vmoe4 --rm -it -v $(pwd)/app:/home/user/app -v $tfds $workdir $image zsh"
elif [ "$2" == "notebook" ]; then
	if [ $# -ne 3 ]; then
		echo "Jupyter notebook's port is 7777."
		port=7777
	else
		port=$3
	fi
	eval "docker run $gpu --name vmoe_notebook --rm -it -v $(pwd)/app:/home/user/app -v $tfds -p $port:$port $workdir $image zsh -c 'cd app && /home/user/.local/bin/jupyter notebook --ip 0.0.0.0 --port=$port'"
elif [ "$2" == "tensorboard" ]; then
	if [ $# -ne 3 ]; then
		echo "Tensorboard's port is 7778."
		port=7778
	else
		port=$3
	fi
	eval "docker run $gpu --name vmoe_tensorboard --rm -it -v $(pwd)/app:/home/user/app -v $tfds -p $port:7777 $image zsh -c 'cd app && ./tensorboard.sh'"
fi
