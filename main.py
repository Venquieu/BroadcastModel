import numpy as np

from model import Model
from searcher import Searcher
from utils import get_distance

def get_random_num(k, b, dim0=30, dim1=2):
    return k * np.random.rand(dim0, dim1) + b

def get_center(num=1):
    x0 = np.random.rand(num, 1) / 100 + 39.78
    y0 = np.random.rand(num, 1) / 100 + 116.22
    power_init = 30 * np.random.rand(num, 1) + 110  # range:(110,140)
    frequency = np.array(num * [100]).reshape(num, 1)  # 100

    center_info = np.array([x0, y0, frequency, power_init]).reshape(num, 4)
    return center_info

def get_point(center_info, axis_limit=1):
    NUM_POINTS = 30
    frequency = np.array(NUM_POINTS * [100])
    sites = np.array(get_random_num(axis_limit, 0))  # (30,2) range:0-1
    sites = sites / 100 + np.array([39.78, 116.22])
    centers = center_info[:, :2]
    center_num = centers.shape[0]
    dist = get_distance(centers, sites)

    generater_model = Model()
    reduction = generater_model.propagation(dist, methods="Okumura_Hata")
    # print('dist&reduction:',[dist,reduction])
    center_power = center_info[:, -1:]
    point_power = np.maximum(center_power - reduction, 0)  # (m,30)
    point_power = np.sum(point_power, axis=0)

    point_info = np.array([sites[:, 0], sites[:, 1], frequency, point_power]).T
    return point_info


def data_generator(source_num=1):
    center_info = get_center(source_num)
    point_info = get_point(center_info)
    return center_info, point_info

def main(data, test_site=None, real_center=None):
    """
    data:       输入数据-->(latitude,longtitude,frequency,power)*N
    test_site:  待求点的坐标，输出该点能量强度,支持多点输入
    """
    searcher = Searcher(
        data,
        variance_threshold=20,
        alpha=1e-5,
        center_num=1,
        center_movement_allow=5e-4,
        scale=2,
    )
    searcher.find_all_source(epochs=100, visualization=True, real_center=real_center)
    print("最终发射源坐标预测：", searcher.centers)

    if test_site is not None:
        power = searcher.predict_point_power(test_site)  # 获取测试点能量
        print('中心点能量强度：', searcher.center_power)
        print("测试点能量强度：", power)


if __name__ == "__main__":
    center_info, dataset = data_generator(source_num=2)
    print("real_center_info:", center_info)

    main(dataset, real_center=center_info)
