mode=${1}
config=${2}
config2=$(echo $config | sed 's/\.[^.]*$//')
replace=${config2//'/'/'.'}
sub=${replace:9}
output="./output/"$sub
echo "Saving to Output directory: "$output
curtime=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: "$curtime
if [ "$mode" = "train" ]
then
echo $mode
python3 projects/Distillation/train_net.py --config-file=$config \
                                           --num-gpus=4 \
                                           --resume \
                                           --dist-url=tcp://127.0.0.1:$RANDOM \
                                           OUTPUT_DIR $output \
                                           TEST.EVAL_PERIOD 0
fi


if [ "$mode" = "debugtrain" ]
then 
echo $mode
python3 projects/Distillation/train_net.py --config-file=$config \
	                                   --num-gpus=1 \
					                           OUTPUT_DIR $output".debug" \
					                           SOLVER.IMS_PER_BATCH 1
fi

if [ "$mode" = "debugtrain2" ]
then
echo $mode
python3 projects/Distillation/train_net.py --config-file=$config \
	                                   --num-gpus=2 \
					                           OUTPUT_DIR $output".debug" \
					                           SOLVER.IMS_PER_BATCH 4
fi

if [ "$mode" = "eval" ]
then
echo $mode
python3 projects/Distillation/train_net.py --config-file=$config \
	                                   --num-gpus=2 \
	                                   --eval-only \
                                     OUTPUT_DIR $output".eval" \
                                     MODEL.WEIGHTS $output"/model_final.pth" \
                                     SOLVER.IMS_PER_BATCH 4
fi

curtime=$(date "+%Y-%m-%d %H:%M:%S")
echo "End time: "$curtime
