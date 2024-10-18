

command_file=`basename "$0"`
script_file=gen_pc_with_label.py
gpu=0,1
data_root=/mnt/d/Synthia4D/syn
output_dir=/mnt/d/Synthia4D/preprocessed
camera_name=Stereo_Left
npoint=16384
downsample_rate=16
debug=0
log_dir=log_gen_pc_with_label
log_file=$log_dir.txt

python3 $script_file \
    --gpu $gpu \
    --data_root $data_root \
    --output_dir $output_dir \
    --camera_name $camera_name \
    --npoint $npoint \
    --downsample_rate $downsample_rate \
    --debug $debug \
    > $log_file 2>&1 &
