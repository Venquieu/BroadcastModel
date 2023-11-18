import numpy as np

def degree2radian(degree):
    return degree * np.pi / 180.0

def get_distance(site0, site1):
    """
    由经纬度坐标得到实际距离\n
    site0：中心点坐标，(M,2)\n
    site1：监测点坐标，(N,2)
    """
    centers_num = site0.shape[0]
    dist = []
    for i in range(centers_num):
        center = site0[i, :]
        radlat1 = degree2radian(center[0])
        radlat2 = degree2radian(site1[:, 0])
        a = radlat1 - radlat2  # (N,)
        b = degree2radian(center[1]) - degree2radian(site1[:, 1])  # (N,)
        s = 2 * np.arcsin(
            np.sqrt(
                np.sin(a / 2) ** 2
                + np.cos(radlat1) * np.cos(radlat2) * np.sin(b / 2) ** 2
            )
        )
        s = s * 6378.137  # (N,)
        dist.append(s)
    dist = np.array(dist).reshape(centers_num, site1.shape[0])  # (m,N)
    return dist
