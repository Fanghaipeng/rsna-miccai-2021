# RSNA-MICCAI-2021


## Usage
命令行bash train.sh后，运行main.py函数进行训练。  
train.sh的相关参数如下：  

- config  
此次训练对应的config文件

- workers  
好像是多线程相关的参数？DataLoader里面用到

- gpu_id and gpu_num  
单卡或多卡训练；（其中多卡训练时暂时使用的还是DataParallel，还没改成DistributeDataParallel）
train.sh中gpu_id和gpu_num的使用要求为
```
--gpu_id 0  --gpu_num 1 # 仅使用一张卡,适用于单卡/多卡仅使用编号为0的卡
--gpu_id 1  --gpu_num 1 # 仅使用一张卡,适用于多卡使用编号为1
--gpu_id -1 --gpu_num 2 # 使用两张卡,gpu_id必须为-1,且多卡编号必须为从0开始的连续值
```
