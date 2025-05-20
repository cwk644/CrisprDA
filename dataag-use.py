# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
import math
from utils import *
import time
from sklearn.neighbors import KernelDensity
def conduct(x):
    a=0
    b=0
    c=0
    for i in x:     
        if (i<0.25):
            a=a+1
        elif (i>0.75):
            c=c+1
        else:
            b=b+1
    return [a,b,c]

def get_mixup_sample_rate(data_list,y_list, kernel="gaussian",bandwidth=1.0):
    data_list=data_list.reshape(data_list.shape[0],-1)
    mix_idx = []
    N = len(data_list)
    
    index=[]
    ######## use kde rate or uniform rate #######
    for i in range(N):
        data_i = data_list[i]
        data_i = data_i.reshape(-1,data_i.shape[0])
        kd = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data_i)
        each_rate = np.exp(kd.score_samples(data_list))
        each_rate /= np.sum(each_rate)  
        tmp=wt_sample(each_rate)
        index.append(tmp)
        if (i%1000==0):
            print("Now i=",i)
    index=np.array(index)
    #self_rate = [mix_idx[i][i] for i in range(len(mix_idx))]

    return index



def weight_sampling(w_list):
    ran = np.random.uniform(0,1)
    sum=0
    for i in range(len(w_list)):
        sum+=w_list[i]
        if(ran<sum):
            return i

def wt_sample(w_list):
    tmp=weight_sampling(w_list)
    return np.array(tmp)


def C_mixup(input_x, input_y, alpha):
    # get mixup lambda
    #input_y = tf.one_hot(input_y, depth=label_size)
    
    index=get_mixup_sample_rate(input_y, input_y)
    #index=wt_sample(mdx)
    
    mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    mix = tf.maximum(mix, 1 - mix)

    # get random shuffle sample
    random_x = tf.gather(input_x, index)
    random_y = tf.gather(input_y, index)
    mix=tf.cast(mix,tf.double)
    # get mixed input
    xmix = tf.cast(input_x,tf.float64) * mix + tf.cast(random_x,tf.float64) * (1 - mix)
    ymix = tf.cast(input_y, tf.float64) * mix + tf.cast(random_y, tf.float64) * (1 - mix)
    return xmix, ymix

def compare(a,b):
    dis=np.linalg.norm(a-b,ord=2)
    return dis

def min_max_normalize(arr):
    min_val = min(arr)
    max_val = max(arr)
    normalized_arr= (arr-min_val)/(max_val-min_val)
    #normalized_arr = [(x - min_val) / (max_val - min_val) for x in arr]
    return normalized_arr

def C_mixup_noretry(input_x, input_y, alpha,index):
    # get mixup lambda
    #input_y = tf.one_hot(input_y, depth=label_size)
    
    #index=get_mixup_sample_rate(input_y, input_y)
    #index=wt_sample(mdx)
    
    mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    mix = tf.maximum(mix, 1 - mix)

    # get random shuffle sample
    random_x = tf.gather(input_x, index)
    random_y = tf.gather(input_y, index)
    mix=tf.cast(mix,tf.double)
    # get mixed input
    xmix = tf.cast(input_x,tf.float64) * mix + tf.cast(random_x,tf.float64) * (1 - mix)
    ymix = tf.cast(input_y, tf.float64) * mix + tf.cast(random_y, tf.float64) * (1 - mix)
    return xmix, ymix

def Add_bio_noise_controlled(x, max_mutations=2, non_seed_indices=None, transition_prob=0.9):
    """
    对输入的one-hot编码序列 x (形状: (batch_size, 23, 4))
    在指定的非种子区域（non_seed_indices）中，每个样本最多随机选择max_mutations个位置进行碱基替换，
    替换时优先采用转换 (transition)，仅以小概率使用颠换 (transversion)。
    
    参数：
      x: numpy数组或tf.Tensor，形状 (batch_size, 23, 4)
      max_mutations: 每个样本最大允许突变的个数（默认2）
      non_seed_indices: 允许发生突变的序列位置列表，例如如果认为后面部分为关键区域，则可以设定为list(range(12))，只对前12个位置进行扰动
      transition_prob: 使用转换的概率（如0.8表示80%的概率使用转换，其余则采用颠换）
                          
    返回：
      x_noisy: 与 x 形状相同的 one-hot 编码数组，经过受控突变后的版本
    """
    # 确保 x 为 numpy 数组
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    batch_size, seq_len, num_bases = x.shape
    if non_seed_indices is None:
        non_seed_indices = list(range(seq_len))
    
    # 将one-hot转换为整数表示，形状 (batch_size, seq_len)
    x_int = np.argmax(x, axis=-1)
    
    # 定义转换和颠换映射（假设one-hot顺序为 [A, T, C, G]，对应整数 0,1,2,3）
    # 转换：A↔G, T↔C
    transition_map = {0: 3, 3: 0, 1: 2, 2: 1}
    # 颠换：其他可能的替换（例如，对于A（0），颠换为T（1）或C（2））
    transversion_map = {
        0: [1, 2],  # A -> [T, C]
        1: [0, 3],  # T -> [A, G]
        2: [0, 3],  # C -> [A, G]
        3: [1, 2]   # G -> [T, C]
    }
    
    # 对每个样本进行操作
    print("Add bio_noise starts")
    for i in range(batch_size):
        # 随机决定本样本要进行几处突变（0、1或2），但不超过允许的非种子区域数目和max_mutations
        num_mutations = np.random.choice([0, 1, 2])
        num_mutations = min(num_mutations, len(non_seed_indices), max_mutations)
        # 从允许突变的位置中随机选择 num_mutations 个位置
        mutation_positions = random.sample(non_seed_indices, num_mutations)
        for pos in mutation_positions:
            orig = x_int[i, pos]
            if random.random() < transition_prob:
                # 采用转换：直接替换为对应的转换碱基
                new_base = transition_map[orig]
            else:
                # 采用颠换：从其他选项中随机选择一个
                new_base = random.choice(transversion_map[orig])
            x_int[i, pos] = new_base
            
    # 将整数表示转换回one-hot编码，shape: (batch_size, seq_len, num_bases)
    x_noisy = np.eye(num_bases)[x_int]
    print("Add bio_noise end")
    return x_noisy

def augmix_non_seed(input_x, input_y, encoder, decoder, alpha=0.4, seed_start=10):
    """
    参数：
      input_x: 原始输入数据，形状 (N,23,4)
      input_y: 对应标签（这里一般保持不变）
      encoder: 已训练的encoder模型，输入形状 (23,4)，输出形状 (N, latent_dim)
      decoder: 已训练的decoder模型，输入形状 (latent_dim)，输出形状 (23,4)
      alpha: Beta分布参数，用于采样mixup系数
    """
    # 1. 定义非种子区位置：假设非种子区为索引 0 到 seed_start-1
    non_seed_indices = list(range(seed_start))
    x_noise = Add_bio_noise_controlled(input_x,non_seed_indices=non_seed_indices)
    
    
    # 3. 分别通过encoder得到latent表示
    # 这里假设encoder是一个Keras模型，输入形状 (23,4)，输出形状为 (latent_dim,)
    latent_orig = encoder.predict(input_x)    # shape: (N, latent_dim)
    latent_noise = encoder.predict(x_noise)     # shape: (N, latent_dim)
    
    # 4. 对同一条样本的latent做mixup：生成混合latent
    N = latent_orig.shape[0]
    lam = np.random.beta(alpha, alpha, size=(N, 1))  # shape (N,1)
    latent_mix = lam * latent_orig + (1 - lam) * latent_noise  # 线性插值
    
    # 5. 通过decoder还原出增强样本
    xmix = decoder.predict(latent_mix)  # 期望输出形状: (N, 23, 4)
    
    # 6. 标签保持不变
    ymix = input_y
    
    x_noise = np.reshape(x_noise,newshape=(-1,92))
    return x_noise, input_y
    #return xmix, ymix

def latent_weighted_perturb_with_saved_weights(latent, weights, scale_factor=1.0):
    """
    对latent表示 (N, latent_dim) 使用预先计算好的扰动权重进行扰动。
    直接对所有维度按权重加扰动，扰动幅度为：噪声（均值0，标准差 perturb_strength）乘以权重。
    
    参数：
      latent: numpy数组，形状 (N, latent_dim)
      weights: numpy数组，形状 (latent_dim,)，已预先保存，反映各维度扰动权重
      perturb_strength: 控制扰动标准差（例如 0.05）
      
    返回：
      latent_perturbed: 加权扰动后的latent表示，形状同latent
    """
    latent = latent.copy()
    
    batch_std = np.std(latent, axis=0)  # shape: (latent_dim,)
    batch_std = np.clip(batch_std, 1e-5, None)
    N, latent_dim = latent.shape
    noise = np.random.normal(0, 1, size=(N, latent_dim))  
    # 使用标准差和 scale_factor 调整噪声幅度
    noise_scaled = noise * (batch_std * scale_factor)
    # 对每个维度乘以扰动权重
    noise_weighted = noise_scaled * (weights*20)   
    # 别问 问就是之前保存的之后乘了0.05忘记改了
    latent_perturbed = latent + noise_weighted
    return latent_perturbed

def augmix_latent(input_x, input_y, encoder, decoder, 
                                  alpha=0.4, weight_path='seed_latent_weights.npy'):
    """
    对选中的样本（input_x, shape (N,23,4)）执行基于latent空间的加权扰动+mixup增强：
      1. 通过encoder得到latent表示 (N, latent_dim)。
      2. 直接加载预先保存的扰动权重（shape为 (latent_dim,)）。
      3. 对latent表示使用预先保存的权重进行加权扰动。
      4. 对原始latent与扰动latent做mixup（对每个样本采样Beta混合系数）。
      5. 通过decoder还原生成增强样本 (N,23,4)。
    参数：
      input_x: 原始输入数据，形状 (N,23,4)
      input_y: 对应标签（保持不变）
      encoder: 已训练的encoder模型，输入 (23,4)，输出 latent 表示 (latent_dim,)
      decoder: 已训练的decoder模型，输入 latent 表示 (latent_dim)，输出 (23,4)
      alpha: Beta分布参数，用于采样mixup系数
      perturb_strength: 控制扰动标准差（例如 0.05）
      weight_path: 预先保存扰动权重的文件路径（npy文件），例如 'seed_latent_weights.npy'
      
    返回：
      x_aug: 增强后的样本，形状 (N,23,4)
      y_aug: 标签（保持不变）
    """
    # 1. 通过encoder获得原始latent表示
    latent_orig = encoder.predict(input_x)  # shape: (N, latent_dim)
    
    # 2. 加载预先保存的权重
    weights = np.load(weight_path)  # shape: (latent_dim,)
    
    # 3. 对latent进行加权扰动
    latent_noise = latent_weighted_perturb_with_saved_weights(latent_orig, weights)
    
    # 4. 对同一条样本的latent做mixup：利用Beta分布采样混合系数
    N = latent_orig.shape[0]
    lam = np.random.beta(alpha, alpha, size=(N, 1))  # 每个样本一个系数
    latent_mix = lam * latent_orig + (1 - lam) * latent_noise
    
    # 5. 通过decoder生成增强样本
    x_aug = decoder.predict(latent_mix)  # 形状: (N, 23, 4)
    y_aug = input_y
    return x_aug, y_aug

def c_mixup_select_pairs(X, y,max_C, delta=0.1):

    N = len(X)
    # 对 y 排序后，用双指针/邻近方法能加速，但简单起见用 O(N^2) 方案
    # 对大数据可做更高效方法
    pairs = []
    length = 0
    for i in range(N):
        for j in range(i+1, N):
            if (length <= max_C and (abs(y[i] - y[j]) <= delta)):
                pairs.append((i, j))
                length = length + 1
    return pairs

def c_mixup(X, y, pairs, alpha=0.4):
    #普通的mixup
    if len(pairs) == 0:
        # 没有可混对, 返回空增强
        return np.array([]), np.array([])
    new_x_list = []
    new_y_list = []
    for (i, j) in pairs:
        lam = np.random.beta(alpha, alpha)
        # mixup
        x_i, x_j = X[i], X[j]
        y_i, y_j = y[i], y[j]
        x_mix = lam * x_i + (1 - lam) * x_j
        y_mix = lam * y_i + (1 - lam) * y_j
        new_x_list.append(x_mix)
        new_y_list.append(y_mix)
    X_new = np.stack(new_x_list, axis=0)
    y_new = np.array(new_y_list)
    return X_new, y_new

def c_mixup_enhance(X, y, encoder,decoder,max_C,delta=0.1, alpha=0.4):
    
    print("C_mixup_begin")
    x_latent = encoder.predict(X)
    pairs = c_mixup_select_pairs(x_latent, y,max_C,delta=delta)
    X_new, y_new = c_mixup(x_latent, y, pairs, alpha=alpha)
    X_new = decoder.predict(X_new)
    print("C_mixup_end")
    return X_new, y_new

def Automix_three_methods(input_x, input_y, encoder, decoder,
                          # alpha for each method
                          alpha_A=0.4, alpha_B=0.4, alpha_C=0.4,
                          # ratio for each method
                          selection_ratio_A=0.3, 
                          selection_ratio_B=0.3,
                          selection_ratio_C=0.4,
                          seed_start=10, 
                          delta_c=0.1):
    """
    示例: 三种增强方法:
      - 方法A: augmix_latent
      - 方法B: augmix_non_seed
      - 方法C: c_mixup_enhance (latent c-mixup)
    根据 selection_ratio_X 来将输入数据分割成三块,
    分别用三种增强方法, 最终合并返回.

    参数:
      input_x, input_y: 原数据
      encoder, decoder: 分别用来在latent空间处理
      alpha_A, alpha_B, alpha_C: 对应三种方法的 mixup/beta分布参数
      selection_ratio_X: 各方法的比例
      delta_c: c_mixup_enhance 的活性差阈值

    返回:
      new_x, new_y: 增强后的合并数据
    """
    N = input_x.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    # 计算每种方法要分到多少数据
    num_A = int(selection_ratio_A * N)
    num_B = int(selection_ratio_B * N)
    num_C = int(selection_ratio_C * N)
    if num_A + num_B + num_C > N:
        # 如果三者之和 > N, 可以截断 or 调整
        num_C = N - (num_A + num_B)

    idx_A = indices[:num_A]
    idx_B = indices[num_A:num_A+num_B]
    idx_C = indices[num_A+num_B : num_A+num_B+num_C]
    remaining_idx = indices[num_A+num_B+num_C:]
    
    if (num_A == 0 and num_C == 0):
        x_B = input_x[idx_B]
        y_B = input_y[idx_B]
        x_aug_B, y_aug_B = augmix_non_seed(x_B, y_B, encoder, decoder, alpha=alpha_B, seed_start=seed_start)
        new_x = np.reshape(x_aug_B,newshape=(-1,23,4))
        new_y = y_aug_B
        print(f"methodA enhanced  {x_aug_B.shape[0]} ")
        return new_x,new_y
    # A 数据
    x_A = input_x[idx_A]
    y_A = input_y[idx_A]
    x_aug_A, y_aug_A = augmix_latent(x_A, y_A, encoder, decoder, alpha=alpha_A)
    # B 数据
    x_B = input_x[idx_B]
    y_B = input_y[idx_B]
    x_aug_B, y_aug_B = augmix_non_seed(x_B, y_B, encoder, decoder, alpha=alpha_B, seed_start=seed_start)

    # C 数据 (c_mixup_enhance)
    x_C = input_x[idx_C]
    y_C = input_y[idx_C]
    x_aug_C, y_aug_C = c_mixup_enhance(x_C, y_C, encoder, decoder, delta=delta_c,max_C = num_C, alpha=alpha_C)
    
    new_x = np.concatenate([x_aug_A, x_aug_B, x_aug_C], axis=0)
    new_y = np.concatenate([y_aug_A, y_aug_B, y_aug_C], axis=0)
    new_x = np.reshape(new_x,newshape=(-1,23,4))
    
    print(f"methodA enhanced {x_aug_A.shape[0]} | methodB enhanced {x_aug_B.shape[0]} | methodC enhanced {x_aug_C.shape[0]} => total new={new_x.shape[0]}")
    return new_x, new_y


def compute_seed_latent_importance(encoder, decoder, input_x, seed_output_indices):
    """
    计算每个 latent 维度对 decoder 输出中种子区域的重要性，
    种子区域由 seed_output_indices 指定（例如 [12, 13, ..., 22]）。
    
    参数：
      encoder: 已训练的 encoder 模型，输入 (23,4)，输出 latent 表示 (latent_dim,)
               这里我们假设使用 encoder 的 z_mean 作为 latent 表示。
      decoder: 已训练的 decoder 模型，输入 latent 表示 (latent_dim)，输出 (23,4)
      input_x: 输入数据，形状 (N, 23, 4)
      seed_output_indices: 一个列表，指定 decoder 输出中哪些位置属于种子区，例如 list(range(12, 23))
      
    返回：
      importance_avg: 一个 numpy 数组，形状 (latent_dim,), 表示每个 latent 维度对种子区域输出的重要性，
                      数值越大说明该维度对种子区贡献越大。
    """
    print("Compute seed latent importance starts:")
    input_tensor = tf.convert_to_tensor(input_x, dtype=tf.float32)
    N = input_x.shape[0]
    importance_list = []
    
    for i in range(N):
        x_i = input_tensor[i:i+1]  # shape: (1, 23, 4)
        if i % 100 == 0:
            print(f"Processing sample {i}")
        with tf.GradientTape() as tape:
            # 获取 latent 表示
            latent = encoder(x_i)  # 假设 latent 的形状为 (1, latent_dim)
            tape.watch(latent)
            # 解码得到输出
            decoded = decoder(latent)  # 输出形状: (1, 23, 4)
            # 提取种子区的输出（例如，假设种子区对应输出的索引 12~22）
            seed_out = tf.gather(decoded, indices=seed_output_indices, axis=1)  # shape: (1, len(seed_output_indices), 4)
            # 定义目标函数：这里简单取种子区输出的总和
            target = tf.reduce_sum(seed_out)
        grads = tape.gradient(target, latent)  # 形状: (1, latent_dim)
        importance_list.append(tf.abs(grads)[0].numpy())
    
    importance_avg = np.mean(np.stack(importance_list, axis=0), axis=0)  # shape: (latent_dim,)
    print("Compute seed latent importance done.")
    return importance_avg

def save_seed_latent_perturbation_weights(encoder, decoder, input_x, seed_output_indices, base_strength=0.05, min_factor=0.0, save_path='seed_latent_weights.npy'):
    """
    根据 encoder 和 decoder 对输入数据进行特征重要性分析，计算每个 latent 维度对 decoder 输出中种子区域的贡献，
    并根据归一化的重要性计算扰动权重：
         weight = base_strength * (1 - normalized_importance) + min_factor
    这样，种子区贡献越大的latent维度扰动权重越低。
    最后保存这个权重向量到 npy 文件中。
    
    参数：
      encoder: 已训练的encoder模型，输入 (23,4)，输出 latent 表示 (latent_dim,)
      decoder: 已训练的decoder模型，输入 latent 表示 (latent_dim)，输出 (23,4)
      input_x: 用于归因计算的数据，形状 (N,23,4)
      seed_output_indices: 指定decoder输出中哪些位置属于种子区，例如 list(range(12, 23))
      base_strength: 基础扰动强度
      min_factor: 最小扰动因子
      save_path: 保存文件路径
    返回：
      weights: 一个 numpy 数组，形状 (latent_dim,), 表示扰动权重
    """
    # 计算每个latent维度的重要性
    importance = compute_seed_latent_importance(encoder, decoder, input_x, seed_output_indices)

    
    # 对重要性进行归一化到 [0, 1]
    importance_min = np.min(importance)
    importance_max = np.max(importance)
    norm_importance = (importance - importance_min) / (importance_max - importance_min)
    # 定义扰动权重：重要性越高，扰动越低
    gamma = 2
    weights = base_strength * ((1 - norm_importance) ** gamma)
    print("Raw importance:", weights, np.max(weights),np.min(weights))
    np.save(save_path, weights)
    print(f"Saved seed latent perturbation weights (shape {weights.shape}) to {save_path}")
    return weights

def filter_middle(train_x,train_y,min_score = 0.2,max_score = 0.7):
    if (train_x.shape[0]!=train_y.shape[0]):
        print("shape of train_x and train_y different!")
        return train_x,train_y
    index = []
    for i in range(0,train_x.shape[0]):
        if (min_score<train_y[i] and train_y[i]<max_score):
            index.append(i)
    
    return train_x[index],train_y[index]


def automix_one_part(input_x, input_y, encoder, decoder,
                          alpha=0.4,selection_ratio=0.3, part = "A",
                          seed_start=10,delta_c=0.1):

                     
    N = input_x.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    num = int(selection_ratio * N)
    idx = indices[:num]

    x_idx = input_x[idx]
    y_idx = input_y[idx]
    
    if (part=="A"):
        new_x, new_y = augmix_latent(x_idx, y_idx, encoder, decoder, alpha=alpha)
    elif (part=="B"):
        new_x, new_y = augmix_non_seed(x_idx, y_idx, encoder, decoder, alpha=alpha, seed_start=seed_start)
    else:
        new_x, new_y = c_mixup_enhance(x_idx, y_idx, encoder, decoder, delta=delta_c,max_C = num, alpha=alpha)
        
    new_x = np.reshape(new_x,newshape=(-1,23,4))
    print(f"part {part}, total new={new_x.shape[0]}")
    return new_x, new_y
                     
def Add_noise(x):
    lens=tf.shape(x)[1]
    noise_add=np.zeros(shape=(lens,))
    k=0.1
    for i in range(lens):
        noise_add[i]=random.uniform(1-k,1+k)
    x=x*noise_add
    return x

def int_array(x):
    x2=np.array(x)
    shape0=x.shape[0]
    shape1=x.shape[1]
    for i in range(shape0):
        for j in range(shape1):
            if (x2[i][j]<0.5):
                x2[i][j]=0
            else:
                x2[i][j]=1
    return x2

def augmix(input_x,input_y,before,after,alpha):
    batch_size = tf.shape(input_x)[0]
    lens=tf.shape(input_x)[1]
    
    random_x_middle=before.predict(input_x)
    random_x_middle=Add_noise(random_x_middle)
    random_x=after.predict(random_x_middle)
    random_x=np.reshape(random_x,newshape=(-1,23,4))
    mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    mix = tf.maximum(mix, 1 - mix)
    
    t_middle=before.predict(input_x)
    t_middle=mix * t_middle + (1-mix) * (before.predict(random_x))
    xmix=after.predict(t_middle)
    ymix=input_y

    return xmix,ymix

def augmix_revise(input_x,input_y,before,after,alpha):
    batch_size = tf.shape(input_x)[0]
    lens=tf.shape(input_x)[1]
    
    original_x=input_x
    
    k=2 #epochs
    xmix=[]
    ymix=[]
    for i in range(k):
        random_x_middle=before.predict(input_x)
        random_x_middle=Add_noise(random_x_middle)
        '''
        random_x=after.predict(random_x_middle)
        random_x=np.reshape(random_x,newshape=(-1,23,4))
        '''
        mix = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
        mix = tf.maximum(mix, 1 - mix)

        t_middle=before.predict(original_x)
        t_middle=mix * t_middle + (1-mix) * (random_x_middle)
        input_x=after.predict(t_middle)
        input_x=tf.reshape(input_x,shape=(-1,23,4))
        xmix.append(input_x)
        ymix.append(input_y)
    #xmix=int_array(xmix)
    
    xmix=np.concatenate(xmix)
    ymix=np.concatenate(ymix)

    return xmix,ymix          