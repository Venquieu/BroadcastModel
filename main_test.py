#import broadcast_model as bm
from Model import model
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

def get_random_num(k,b,dim0 = 30,dim1 = 2):
    return k*np.random.rand(dim0,dim1)+b

generater_model = model()

def rad(degree):
    return degree*np.pi/180.0

def GetDistance(site0,site1):
    '''
    由经纬度坐标得到实际距离\n
    site0：中心点坐标，(m,2)\n
    site1：监测点坐标，(N,2)或(N,4)
    '''
    centers_num = site0.shape[0]
    dist = []
    for i in range(centers_num):
        center = site0[i,:]
        radlat1 = rad(center[0])
        radlat2 = rad(site1[:,0])
        a = radlat1 - radlat2 #(N,)
        b = rad(center[1]) - rad(site1[:,1]) #(N,)
        s = 2 *np.arcsin(np.sqrt(np.sin(a/2)**2+np.cos(radlat1)*np.cos(radlat2)*np.sin(b/2)**2))
        s = s*6378.137 #(N,)
        dist.append(s)
    dist = np.array(dist).reshape(centers_num,site1.shape[0]) #(m,N)
    return dist

def get_center(num = 1): 
    x0 = np.random.rand(num,1) #range:(0,1)
    y0 = np.random.rand(num,1) #range:(0,1)
    power0 = 30*np.random.rand(num,1)+110 #range:(110,140)
    frequency = np.array(num*[100]).reshape(num,1)#100
    center_info = np.array([x0,y0,frequency,power0]).reshape(4,num)
    center_info = center_info.T
    center_info[:,:2] = center_info[:,:2]/100 +np.array([39.78,116.22])
    #print('center_info',center_info)
    return center_info

def get_point(center_info,axis_limit = 1):
    #center_info = get_center(3)
    point_num = 30
    frequency = np.array(point_num*[100])
    sites = np.array(get_random_num(axis_limit,0)) #(30,2) range:0-1
    sites = sites/100+np.array([39.78,116.22])
    centers = center_info[:,:2]
    center_num = centers.shape[0]
    dist = GetDistance(centers,sites)
    '''centers = centers.reshape(center_num,1,2) #(m,1,2)
    dist = np.linalg.norm(centers - sites,axis=2,keepdims=True)
    dist = dist.reshape(center_num,point_num) #(m,N) 每个监测点距每个中心点的距离'''

    reduction = generater_model.propagation(dist,methods='Okumura_Hata')
    #print('dist&reduction:',[dist,reduction])
    center_power = center_info[:,-1].reshape(center_num,1)
    point_power = np.maximum(center_power - reduction,0)  #(m,30)
    point_power = np.sum(point_power,axis=0)
    #point_power = point_power.reshape(30)
    point_info = np.array([sites[:,0],sites[:,1],frequency,point_power])
    point_info = point_info.T 
    #print('point_info',point_info)
    return point_info

def visual(center_info,point_info,scale = 1,axis_limit = 40):
    #plt.xlim((0,axis_limit))
    #plt.ylim((0,axis_limit))
    plt.scatter(point_info[:,0],point_info[:,1],s = point_info[:,-1]/scale)#监测点散点图
    plt.scatter(center_info[:,0],center_info[:,1],s = center_info[:,-1]/scale)#中心点散点图
    plt.show()

def data_factory(source_num = 1):
    if source_num == 1:
        data = np.array([[39.79310526,116.25213005,100.0,58.793080543949884],
        [39.78828712,116.22224312,100.0,43.35031684684918],
        [39.79222208,116.23649466,100.0,53.199376460682785],
        [39.81377104,116.25588055,100.0,56.10655775313832],
        [39.80986538,116.23372063,100.0,54.51493271852607],
        [39.79927903,116.24423895,100.0,71.4355445723539],
        [39.81229124,116.22589538,100.0,47.579393157080915],
        [39.81642161,116.23425904,100.0,49.869840294517815],
        [39.81889211,116.23191275,100.0,47.01618143045897],
        [39.7996801,116.24178395,100.0,67.79669481473167],
        [39.79044932,116.24737529,100.0,55.27209777741899],
        [39.78312236,116.23123005,100.0,44.101198873226025],
        [39.79815778,116.25529185,100.0,65.24470381680374],
        [39.78207808,116.22961185,100.0,42.996175687525195],
        [39.80010361,116.25228068,100.0,74.79198395683457],
        [39.81642058,116.24283575,100.0,53.49713446147763],
        [39.8157906,116.23427732,100.0,50.37630924232067],
        [39.81022262,116.24779516,100.0,65.29032288560265],
        [39.7961952,116.23522142,100.0,55.757678191884665],
        [39.80323219,116.24632093,100.0,90.50407595063986],
        [39.78746475,116.22643602,100.0,44.78403046741789],
        [39.80869698,116.2567423,100.0,62.380588611961684],
        [39.79889665,116.23680803,100.0,59.548016768418094],
        [39.79860634,116.25876509,100.0,61.14968730568006],
        [39.81771563,116.2217318,100.0,43.20496145173459],
        [39.79371572,116.22694502,100.0,48.14063553343203],
        [39.8192644,116.22252362,100.0,42.89420664337979],
        [39.7852034,116.25843138,100.0,47.77795504208953],
        [39.80213355,116.24331609,100.0,74.98786547521944],
        [39.80824291,116.23860975,100.0,60.89703620722747]])
    elif source_num == 2:
        data = np.array([[39.79310526,116.25213005,100.0,92.4704775953801],
        [39.78828712,116.22224312,100.0,67.73448298877587],
        [39.79222208,116.23649466,100.0,86.53365327954394],
        [39.81377104,116.25588055,100.0,91.31604719606221],
        [39.80986538,116.23372063,100.0,96.17651992617739],
        [39.79927903,116.24423895,100.0,119.64062481653013],
        [39.81229124,116.22589538,100.0,79.63017764880186],
        [39.81642161,116.23425904,100.0,84.43666523145542],
        [39.81889211,116.23191275,100.0,77.93769633285697],
        [39.7996801,116.24178395,100.0,116.80233588838246],
        [39.79044932,116.24737529,100.0,87.41576245256418],
        [39.78312236,116.23123005,100.0,68.10933486498197],
        [39.79815778,116.25529185,100.0,102.17242765174348],
        [39.78207808,116.22961185,100.0,65.97597687275835],
        [39.80010361,116.25228068,100.0,117.0403702642287],
        [39.81642058,116.24283575,100.0,91.05307100863712],
        [39.8157906,116.23427732,100.0,85.66763025240064],
        [39.81022262,116.24779516,100.0,113.92073689789721],
        [39.7961952,116.23522142,100.0,93.31573971080748],
        [39.80323219,116.24632093,100.0,151.06108642228781],
        [39.78746475,116.22643602,100.0,70.27735017524037],
        [39.80869698,116.2567423,100.0,100.60376646049379],
        [39.79889665,116.23680803,100.0,102.38353592403784],
        [39.79860634,116.25876509,100.0,95.32181802468841],
        [39.81771563,116.2217318,100.0,69.77304858516608],
        [39.79371572,116.22694502,100.0,78.04324379952818],
        [39.8192644,116.22252362,100.0,69.05179892435352],
        [39.7852034,116.25843138,100.0,72.50617354485121],
        [39.80213355,116.24331609,100.0,134.47447806398083],
        [39.80824291,116.23860975,100.0,113.25020479721593]])
    data = data.reshape(30,4)
    return data

def data_generator(source_num = 1):
    center_info = get_center(source_num)
    point_info = get_point(center_info)
    return center_info,point_info

if __name__ == "__main__":
    center_info = get_center(2)
    point_info = get_point(center_info)
    print('center_info',center_info)
    print('point_info',point_info)
    visual(center_info,point_info)

