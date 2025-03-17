project_dir=/home/luotian.ding/myproject/rectified_flow
export PYTHONPATH=$project_dir:$PYTHONPATH
main_path=${project_dir}/main/experiment.py
cfg_path=${project_dir}/main/config.py
python $main_path $cfg_path