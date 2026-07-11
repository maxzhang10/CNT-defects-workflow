#用于构建缺陷
from ase import Atom

from numpy.linalg import norm
from ase.io import write
import numpy as np
from ase.build import sort

def find_nearest(tube, ind1, only_C=True):
    pos_all = tube.positions
    symbols = np.array(tube.get_chemical_symbols())
    if symbols[ind1 - 1] != "C":
        raise ValueError(f"目标原子 {ind1} 不是 C 原子，无法构造 CNT 缺陷。")
    
    
    pos1 = tube.positions[ind1 - 1]
    
    
    # 去掉自己
    mask = np.ones(len(pos_all), dtype=bool)
    mask[ind1 - 1] = False
    # 计算所有原子的距离 ,pos_all[mask]——形状从 (N,3) 变成 (N-1,3)。
    # np.linalg.norm 意义是 用 NumPy 的线性代数子模块的范数函数
    
    # 找到最近的三个。np.argsort(dists)[:3]将dists按数值大小排序后 取最小的前三个
    # np.where(mask)[0]的0是为了把元组中的数组提取出来

    if only_C:
        mask &= (symbols == "C")

    
    candidate_indices = np.where(mask)[0]   
    dists = np.linalg.norm(pos_all[mask] - pos1, axis=1)   

    if len(candidate_indices) < 3:
        raise ValueError(
            f"目标原子 {ind1} 周围可用的 C 近邻不足 3 个。"
        ) 
    order = np.argsort(dists)[:3]             #argsort指从小到大排序并且返回索引
    nearest_idx = candidate_indices[order]        # 映射回原始索引
    nearest_dist = dists[order]

    return list(zip(nearest_dist[:3], nearest_idx))

#在tube中序号为ind2的C原子附近加3个H原子,H原子位置在ind2和3个近邻原子的连线上，距离近邻原子固定距离 可修改距离
def add_Hatom(tube, ind2):
    new_tube = tube.copy()
    ind = ind2 - 1
    pos1 = new_tube.positions[ind]
    nearest = find_nearest(new_tube, ind2)
    for i in range(3):
        neighbor_index = nearest[i][1]
        pos_neighbor = new_tube.positions[neighbor_index]
        direction = pos1 - pos_neighbor
        direction /= np.linalg.norm(direction)
        Hpos = pos_neighbor + 0.7 * direction
        new_tube.append(Atom('H', Hpos))
    return new_tube

#用于删除tube中序号为ind3的C原子
def delete_atoms(tube,ind3):
    ind = ind3-1
    new_tube = tube.copy()
    del new_tube[ind]
    return new_tube

#用于删除tube中序号为ind4和ind5的C原子，构造DV缺陷
def delete_DV(tube, ind4, ind5):
    new_tube = tube.copy()
    # 永远从大索引删到小索引，避免任何索引重排风险
    for k in sorted([ind4 - 1, ind5 - 1], reverse=True):
        del new_tube[k]
    return new_tube
def rotate_vector(v, axis, theta):
    """绕任意轴旋转向量 v，右手法则，θ 为弧度"""
    axis = axis / np.linalg.norm(axis)
    v_rot = (v * np.cos(theta) +
             np.cross(axis, v) * np.sin(theta) +
             axis * np.dot(axis, v) * (1 - np.cos(theta)))
    return v_rot

def add_57(tube, ind1, ind2):
    """
    在 tube 中选取两个相邻的 C 原子 (ind1, ind2)，
    绕它们的中点旋转 90° 形成 Stone–Wales (5–7) 缺陷。
    """
    new_tube = tube.copy()
    r1 = new_tube.positions[ind1 - 1]
    r2 = new_tube.positions[ind2 - 1]
    """
    print("旋转前坐标")
    print(r1)
    print(r2)
    """
    center = 0.5 * (r1 + r2)
    bond = r2 - r1
    # CNT 轴向 z
    z_axis = np.array([0.0, 0.0, 1.0])
    # 法向量（与 CNT 壁垂直）
    axis = np.cross(z_axis, bond)
    if np.linalg.norm(axis) < 1e-6:
        raise ValueError("原子键平行于 z 轴，无法定义切面")
    axis = axis / np.linalg.norm(axis)
    # 旋转 90°
    theta = np.pi / 2
    # 分别计算旋转后的坐标
    r1_new = center + rotate_vector(r1 - center, axis, theta)
    r2_new = center + rotate_vector(r2 - center, axis, theta)
    """
    print("旋转后坐标")
    print(r1_new)
    print(r2_new)
    """
    # 更新坐标
    new_tube.positions[ind1 - 1] = r1_new
    new_tube.positions[ind2 - 1] = r2_new

    return new_tube



 #构造 divacancy：删除 index1 (1-based) 和它的第 neighbor_rank 个最近邻
def CNT_DV(tube,index1,neighbor_rank=1):
    #因为在find_nearest已经考虑了坐标序数错位1，此处的index 从1开始！！
    near_list = find_nearest(tube,index1)
    nb0 = near_list[neighbor_rank][1]    #nb0 指的是三个近邻原子里的0号，可以修改neighbor_rank为 1、2 来改变DV缺陷方向
    tube_DV = delete_DV(tube, index1,nb0+1)  #是从0开始计数 所以要加一
    return tube_DV
def CNT_MVH (tube, index2):
    tube_MVH = delete_atoms(add_Hatom(tube, index2), index2) 
    #Atoms.append() 只会把新原子加在列表末尾，所以先加H没问题，不会改变C的索引
    return tube_MVH

def CNT_5775 (tube ,index3,neighbor_rank=0):
    near_list = find_nearest(tube,index3)
    nb0 = near_list[neighbor_rank][1]    #nb0 指的是三个近邻原子里的0号，可以修改neighbor_rank为 1、2
    tube_5775 = add_57(tube, index3,nb0+1)
    return tube_5775



