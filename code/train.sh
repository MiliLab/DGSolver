export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

start_time=$(date +%s)
echo "start_time: ${start_time}"

# TORCH_DISTRIBUTED_DEBUG=DETAIL 
# nohup python train.py > ./train.log 2>&1 & 

nohup python -m torch.distributed.launch --nproc_per_node 4 --use_env train.py  > ./train.log 2>&1 & 

end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

echo "------------------ Final result ------------------"