#!/bin/bash

# usage: ./run.sh "loss=(0.005,0.005,0.0003,0.0) LT=jsd_topk TS=False" ("description~~~" "workdir")

#export JAX_DEBUG_NANS=true
#XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95


if [ $# -lt 3 ];then
    workdir=${1// /_}
else
    workdir=$3
fi
workdir=$1

tmp_workdir=$workdir
counter=1
while [ -e workdir/$tmp_workdir/memory.prof ]
do
    tmp_workdir=${workdir}_\($counter\)
    counter=`expr $counter + 1`
done
workdir=$tmp_workdir
mkdir -p workdir/$workdir

exec 1> >(tee -a workdir/$workdir/app.log)
exec 2> >(tee -a workdir/$workdir/error.log)

python3 main.py --workdir=workdir/$workdir --config=vmoe/configs/vmoe_paper/prc_ilsvrc2012_ft.py
#python3 main.py --workdir=workdir/$workdir --config=vmoe/configs/vmoe_paper/prc_cifar100_ft.py
