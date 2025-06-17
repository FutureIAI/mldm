# mldm
MSA-LDM​：多尺度注意力潜在扩散模型，解决工业正常样本生成问题
MLDAG​：多尺度潜在扩散对抗生成算法，用于高质量对抗样本生成
核心方法
MSA-LDM（正常样本）
使用变分自编码器压缩图像到低维空间
U-Net嵌入多尺度注意力模块（并行多尺度卷积+自注意力）
损失函数：重建损失+感知损失+结构损失
MLDAG（对抗样本）
三阶段噪声调度策略：
结构破坏阶段（SSIM引导）
特征扰动阶段（特征相似性约束）
感知补偿阶段（LPIPS微调）
损失函数：攻击损失+感知损失+相似性损失
实验结果
数据集：
MVTec AD（15类/5354图）
BTAD（3类/2830图）

训练:
$ python -m torch.distributed.launch --nproc_per_node=1 train_net.py --dataset MVTec-AD --class_name bottle
生成:
$ python -m torch.distributed.launch --nproc_per_node=1 sample.py --dataset MVTec-AD
