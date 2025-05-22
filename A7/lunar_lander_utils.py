# import numpy as np
#
# def discretize_state(observation, bins_config=[8,8,6,6,6,4]):
#     """
#     Discretizes the first 6 continuous features of the observation array.
#     Uses ranges and bin counts as suggested in the assignment.
#     The number of points for np.linspace is bins_config[i]-1 to create bins_config[i] distinct intervals/states.
#     """
#     # Define bin edges for each of the 6 features
#     # Ranges based on typical LunarLander values and PDF example structure.
#     # state[0]: x-coordinate, range: -1.0 to 1.0
#     # state[1]: y-coordinate, range: -1.0 to 1.0 (as per PDF text and image)
#     # state[2]: x-velocity (absolute), range: 0.0 to 2.0
#     # state[3]: y-velocity (absolute), range: 0.0 to 2.0
#     # state[4]: angle, range: -1.0 to 1.0
#     # state[5]: angular velocity, range: -2.0 to 2.0
#
#     _bin_edges = [
#         np.linspace(-1.0, 1.0, bins_config[0] - 1),  # x-coordinate
#         np.linspace(-1.0, 1.0, bins_config[1] - 1),  # y-coordinate
#         np.linspace(0.0, 2.0, bins_config[2] - 1),   # abs(x-velocity)
#         np.linspace(0.0, 2.0, bins_config[3] - 1),   # abs(y-velocity)
#         np.linspace(-1.0, 1.0, bins_config[4] - 1),  # angle
#         np.linspace(-2.0, 2.0, bins_config[5] - 1)   # angular velocity
#     ]
#
#     features = [
#         np.digitize(observation[0], _bin_edges[0]),
#         np.digitize(observation[1], _bin_edges[1]),
#         np.digitize(abs(observation[2]), _bin_edges[2]),
#         np.digitize(abs(observation[3]), _bin_edges[3]),
#         np.digitize(observation[4], _bin_edges[4]),
#         np.digitize(observation[5], _bin_edges[5])
#     ]
#     # The assignment asks for discretizing 6 key dimensions.
#     # Leg contact information (observation[6] and observation[7]) is not included here.
#     return tuple(features)
# lunar_lander_utils.py
import numpy as np


def get_discretization_params(scheme_id='original'):
    """根据方案ID返回离散化参数"""
    if scheme_id == 'scheme1':  # 关键特征精细化
        bins_config = [8, 12, 6, 10, 10, 4]  # x, y, vx, vy, ang, ang_vel
        # 特征范围保持标准
        ranges = [
            (-1.0, 1.0),  # x-coordinate
            (-1.0, 1.0),  # y-coordinate
            (0.0, 2.0),  # abs(x-velocity)
            (0.0, 2.0),  # abs(y-velocity)
            (-1.0, 1.0),  # angle (radians)
            (-2.0, 2.0)  # angular velocity (radians/s)
        ]
        custom_edges = None
    elif scheme_id == 'scheme2':  # 非均匀分箱 (示例)
        bins_config = [8, 8, 6, 6, 6, 4]  # 可以保持原始分箱数，但某些特征的边界点非均匀
        ranges = [  # 确保范围与自定义边界点对应
            (-1.0, 1.0),
            (-1.0, 1.0),  # y坐标将使用自定义边界
            (0.0, 2.0),
            (0.0, 2.0),  # y速度将使用自定义边界
            (-1.0, 1.0),
            (-2.0, 2.0)
        ]
        # 为y坐标和y速度定义非均匀边界点 (B个箱需要B-1个内部边界点)
        # 示例：为y坐标创建8个非均匀箱，更关注0附近
        y_pos_edges = np.array([-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6])  # 7个边界点 -> 8箱
        # 示例：为|y速度|创建6个非均匀箱，更关注低速
        abs_vy_edges = np.array([0.1, 0.25, 0.5, 0.8, 1.2])  # 5个边界点 -> 6箱
        custom_edges = {
            1: y_pos_edges,  # y坐标 (索引1)
            3: abs_vy_edges  # |y速度| (索引3)
        }
    else:  # 默认/原始方案 (scheme0)
        bins_config = [8, 8, 6, 6, 6, 4]
        ranges = [
            (-1.0, 1.0), (-1.0, 1.0),
            (0.0, 2.0), (0.0, 2.0),
            (-1.0, 1.0), (-2.0, 2.0)
        ]
        custom_edges = None
    return bins_config, ranges, custom_edges


def discretize_state(observation, scheme_id='original'):
    """
    根据指定的方案ID离散化状态。
    """
    bins_config, ranges, custom_edges = get_discretization_params(scheme_id)

    features = []
    processed_observation = [
        observation[0],
        observation[1],
        abs(observation[2]),
        abs(observation[3]),
        observation[4],
        observation[5]
    ]

    for i in range(6):  # 处理6个选定的特征
        obs_feature = processed_observation[i]
        min_r, max_r = ranges[i]
        num_bins = bins_config[i]

        if custom_edges and i in custom_edges:
            edges = custom_edges[i]
        else:
            # num_bins 个箱体需要 num_bins - 1 个内部边界点
            if num_bins <= 1:  # 至少需要一个箱体
                edges = np.array([])  # 没有内部边界点，所有值都在一个箱
            else:
                edges = np.linspace(min_r, max_r, num_bins - 1, endpoint=False)[1:]  # 取内部点，不含min_r
                if num_bins > 1:
                    edges = np.linspace(min_r, max_r, num_bins - 1)  # 生成 num_bins-1 个分割点
                else:  # num_bins = 1, 只有一个区域
                    edges = np.array([])  # 没有分割点

        digitized_feature = np.digitize(obs_feature, bins=edges)
        features.append(digitized_feature)

    return tuple(features)