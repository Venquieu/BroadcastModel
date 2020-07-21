#
#@version:  V3.0
#@data:     2020.6.25
#@brief:    在V2.0基础上重构代码，支持多个广播发射源的情况；
#           优化原代码
#           修复若干已知bug
#@problem： 未考虑检测到不同频率的情况
#
import numpy as np
import heapq
from matplotlib import pyplot as plt
import time

class model:
    '''传播模型'''
    def __init__(self,launch_frequency = 100,launch_height = 50,receive_height = 5):
        self.f = launch_frequency #MHz
        self.hl = launch_height
        self.hr = receive_height

    def free_propagation(self,d,f = None):
        '''自由传播模型'''
        if f == None:
            f = self.f
        d = np.maximum(d,1)
        return np.maximum(32.45+20*np.log10(f)+20*np.log10(d +1e-6),0)

    def Okumura_Hata(self, d,f = None):
        '''
        频率范围:150MHz-1500MHz,在具体工作中可以进行通当扩展;
        移动接收站天线高度: 1-10m;
        固定发射台天线高度:30-200m;
        传输距离: 1-20km。
        此模型是预测城市及周边地区路径损耗模型,在基准的市区路径传播损耗基础上对其它地区进行修正。
        '''
        '''
        f: 信号发射频率，单位兆赫兹(MHz );
        hl:基地台发射天线距离地面高度，单位米(m);
        hr:移动台天线距离地面高度，单位米(m);
        d:接收天线与发射天线之间的距离，单位千米(km);
        a(hm):是与信号源有关的修正因子;.
        K:用于小型城市郊区的校正因子，市区可忽略;
        Q:开阔区校正因子;
        '''
        if f == None:
            f = self.f
        return 69.55 + 26.16*np.log10(f) - 13.82*np.log10(self.hl) + (44.9 - 6.55*np.log10(self.hl))*np.log10(d)

    def Egli(self,d, f = None):
        '''
        Egli传播模型
        适用传输距离:小于60km;
        通用频率：40 -450MHz (小于1000MHz时也可以使用)，对于预测丘陵地形场强较为准确。
        '''
        if f == None:
            f = self.f
        return 88+40*np.log10(d)- 20*np.log10(self.hl*self.hr)+20*np.log10(f)

    def propagation(self,d, f = None,methods = 'free'):
        '''选择使用的模型
            methods：可选'free' 或 'Okumura_Hata' 或 'Egli'  '''
        if methods == 'free':
            return self.free_propagation(d,f)
        elif methods == 'Okumura_Hata':
            return self.Okumura_Hata(d,f)
        elif methods == 'Egli':
            return self.Egli(d,f)


class broadcast_model:
    def __init__(self,data,variance_threshold = 1e3,alpha = 1e-2,max_source = 10,epochs = 500,\
        center_movement_allow = 0.1,scale = 20,center_num = 1,method = 'free',sample_num = None):
        '''
        alpha:  更新的缩放系数，类似于学习率
        max_source：发射源的最大数量，分裂出的中心数量不能超过该数值
        epochs：查找发射源时的迭代次数
        center_movement_allow(km)：中心点移动距离的最小许用值，小于此值认为找到发射源，结束迭代
        scale:  可视化散点图中点的缩小倍数
        methods：使用的模型,可选'free' 或 'Okumura_Hata' 或 'Egli'
        sample_num：监测点数量，默认根据data自动计算
        '''
        self.__alpha = alpha  #更新的缩放系数，类似于学习率
        self.__max_source = max_source #发射源的最大数量
        self.__epochs = epochs #迭代次数
        self.__center_movement_allow = center_movement_allow#中心点移动距离小于此值认为找到发射源，结束迭代
        self.__scale = scale #可视化散点图中点的缩小倍数
        self.__method = method #使用的模型,可选'free' 或 'Okumura_Hata' 或 'Egli'
        self.__variance_threshold = variance_threshold #增加中心点的方差阈值
        self.__centers_num = center_num#中心点初始数量
        if not sample_num == None:
            self.point_num = sample_num   #样本(监测点)数量 简记为n
        self.data = self.__init_data(data,sample_nums=sample_num)

        self.centers = self.__init_center(self.data)  #(1,2)-->(m,2)
        self.__model = model(launch_frequency = self.data[0,2]) #假设所有点频率相同
        self.center_power = []
        self.__point_power_predict = []

    def __init_data(self,input_data,sample_nums = None):
        '''input_data:(latitude,longtitude,frequency,power)*N -->(N,4)'''
        if sample_nums == None:
            data = np.array(input_data)
            assert len(data.shape) == 2 and data.size % 4 == 0 ,'输入数据格式有误'
            self.point_num = data.size//4
        else:
            data = input_data

        return np.array(data).reshape(self.point_num,4)

    def __init_center(self,data,center_number = None):
        '''将强度最大的监测点作为初始发射中心'''
        if center_number == None:
            center_number = self.__centers_num
        powers = data[:,-1]
        index = heapq.nlargest(center_number,range(len(powers)),powers.take) # (center_number,)
        #max_index = np.argmax(data,axis=0) #[index0,index1,index2,index3]  -->(4,1)
        return data[index,0:2].reshape(center_number,2) #(m,2)

    def __visual(self,scale = 20,pause_time = 0.5):
        '''可视化，绘制散点图
        scale：散点图中点的缩小倍数
        pause_time：当前图像的保持（暂停）时长
        '''
        plt.clf()# 清除之前画的图
        plt.scatter(self.data[:,0],self.data[:,1],s = self.data[:,-1]/scale)#监测点散点图
        plt.scatter(self.centers[:,0],self.centers[:,1],s = self.center_power/scale)#中心点散点图
        plt.pause(pause_time)# 暂停一会儿
        #plt.draw()
        plt.ioff()# 关闭画图的窗口
        
    def __get_variance(self,predict_power):
        '''
        确定发射源位置后计算类方差，variance = sum((predict_val-real_val)**2)/N
        Parameters:
        ------------------
        predict_power:(N,1) array
        '''
        return np.sum((predict_power - self.data[:,-1])**2)/self.point_num

    def __split_centers(self, variance,variance_threshold = None,split_mode = True):
        '''根据方差决定是否执行分裂操作以添加发射源
            
            split_mode ：在满足分裂条件时是否执行分裂，默认为执行
        '''
        if variance_threshold == None:
            variance_threshold = self.__variance_threshold

        if variance >variance_threshold:#大于阈值，需要分裂
            if split_mode == True:
                new_center = self.data[np.random.randint(self.point_num),:2]#随机选择一个监测点作为分裂中心点
                new_center = new_center.reshape(1,2)
                #print('split_center:',self.centers,new_center)
                centers = np.vstack((self.centers,new_center))
                self.centers = np.array(centers).reshape(self.__centers_num+1,2)
                self.__centers_num += 1 #m = m + 1
                #print('----------',self.centers.shape,self.__centers_num)
            else:
                pass
            return True
        return False#不需要分裂

    def get_center_power(self,samples_num = 4):
        '''
        计算监测点与中心的距离-->选取samples_num个最近点作为采样点-->计算衰减量-->
            衰减量与各采样点点强度加和求平均作为中心点强度
        Parameters
        ----------
        samples_num:采样点数量，考虑到多发射源的相互干扰，只选用距离中心最近的samples_num个监测点用于计算
        '''
        centers = self.centers.reshape(self.__centers_num,1,2) #(m,1,2)
        dist = np.linalg.norm(centers - self.data[:,:2],axis=2,keepdims=True)
        dist = dist.reshape(self.__centers_num,self.point_num) #(m,N) 每个监测点距每个中心点的距离
        nearest_point_power = []
        nearest_dist = []
        for center_index in range(self.__centers_num):#对每个中心点操作 self.centers.shape = (m,2)
            dist_temp = dist[center_index,:] #每个监测点距该中心点的距离
            index = heapq.nsmallest(samples_num,range(len(dist_temp)),dist_temp.take) #取距离最小的samples_num个点
            #index:[index0,index1,index2...indexn]
            nearest_dist.append(dist_temp[index]) #samples_num个点的距离
            nearest_point_power.append(self.data[index,-1]) #self.data[list,-1] samples_num个点的强度

            '''center_power = []
            dist = np.linalg.norm(self.centers[center_index] - self.data[:,:2],axis=1,keepdims=True)  #(N,1)
            assert dist.shape == (1,self.point_num)
            dist.reshape(self.point_num,1)
            self.__reduction = self.__model.propagation(dist,methods=self.__method) #(N,1)
            center_power.append(np.mean(self.__reduction+self.data[:,-1])) #a real num'''

        nearest_dist = np.array(nearest_dist).reshape(self.__centers_num,samples_num)
        nearest_point_power = np.array(nearest_point_power).reshape(self.__centers_num,samples_num)

        reduction = self.__model.propagation(nearest_dist,methods=self.__method) #(m,samples_num)
        center_power = np.mean(reduction+nearest_point_power,axis=1,keepdims=True) #(m,1)
        self.center_power = center_power.reshape(self.__centers_num,1)#(m,1)
        return dist  #(m,N)

    def move_to_center(self,point_centers_distance):
        '''计算各点预测值 & 计算中心到各个监测点的方向单位向量-->计算应该移动的方向向量-->矢量加和及缩放-->移动中心

            point_centers_distance:(m,N) 每个监测点到每个中心的距离
        '''
        dist = point_centers_distance
        #print('1:',dist.shape)
        centers_power = self.center_power#(m,1)
        reduction = self.__model.propagation(dist,methods=self.__method)#(m,N)
        point_power_predict = centers_power - reduction #(m,N)各中心点对于各点的能量预测
        point_power_predict_sum = np.sum(point_power_predict,axis=0,keepdims=True)#(1,N) 能量加和
        power_weight = point_power_predict/point_power_predict_sum#(m,N) 能量的权重

        point_power = self.data[:,-1].reshape(1,self.point_num) #(1,N) 真实的能量
        point_power_after_weight = power_weight*point_power#(m,N)真实能量按权重分配

        points_site = self.data[:,:2].reshape(self.point_num,2)#(N,2) 各点坐标
        centers_site = self.centers.reshape(self.__centers_num,1,2)#(m,1,2) 中心点坐标
        direction_vector = (points_site - centers_site) #(m,N,2)
        #print('direction_vector:',direction_vector)
        direction_vector /= (np.linalg.norm(direction_vector,axis=2,keepdims=True)+1e-3) #(m,N,2)/(m,N,1)-->(m,N.2)
        power_bias = (point_power_after_weight - point_power_predict).reshape((self.__centers_num,self.point_num ,1))
        directions = power_bias*direction_vector #(m,N,2)
        direction = self.__alpha*np.sum(directions,axis=1) #(m,2)
        #print('self.centers:',self.centers)
        #print('direction:',direction)
        self.centers += direction
        return self.centers

    def find_present_source(self,epochs = None,center_movement_allow = None,visualization = True,scale = None):
        '''迭代至找到发射源'''
        if center_movement_allow == None:
            center_movement_allow =  self.__center_movement_allow
        if scale == None:
            scale = self.__scale

        for __ in range(epochs):
            previous_center = self.centers#记录当前的中心情况 (m,2)
            dist = self.get_center_power()
            self.move_to_center(dist)
            center_movement = np.linalg.norm(previous_center-self.centers,axis=1,keepdims=True) #经过一次迭代后各中心移动量 (m,1)
            center_movement = center_movement.reshape(self.__centers_num,1) #(m,1)
            if visualization == True:
                self.__visual(scale)
                print('发射源坐标预测：',self.centers)
            if center_movement.max() <center_movement_allow: #中心移动量过小则退出
                break

    def find_all_source(self, variance_threshold = None,max_source_num = None,epochs = None,\
        center_movement_allow = None,visualization = True,scale = None):
        '''迭代寻找发射源-->找到发射源后计算方差-->根据方差判断是否需要添加发射源-->
            不需要即结束，需要则重新迭代寻找发射源'''
        if variance_threshold ==None:
            variance_threshold = self.__variance_threshold
        if max_source_num == None:
            max_source_num = self.__max_source
        if epochs == None:
            epochs = self.__epochs
        if center_movement_allow == None:
            center_movement_allow =  self.__center_movement_allow
        if scale == None:
            scale = self.__scale
        if visualization == True:
            plt.ion()#开启绘图的交互模式

        for i in range(max_source_num): #最多迭代max_source_num次，即最多只能有max_source_num个发射源
            self.find_present_source(epochs=epochs,center_movement_allow=center_movement_allow,\
                visualization=visualization,scale=scale)
            predict_power = self.predict_point_power()#各监测点能量
            variance = self.__get_variance(predict_power)
            if not i == max_source_num -1 and not \
                self.__split_centers(variance,variance_threshold=variance_threshold):#迭代结束或不再需要分裂
                break
        if i == max_source_num-1 and self.__split_centers(variance,variance_threshold=variance_threshold,\
            split_mode=False):#迭代了max_source_num次仍未达到要求
            print('未成功找到符合要求的发射源')


    def predict_point_power(self, site = None,method = None):
        '''计算给定坐标的能量强度，用于计算监测点能量和绘制热力图'''
        if np.all(site == None):
            site = self.data[:,:2] #全部监测点s
        if method == None:
            method = self.__method

        site = np.array(site)#(k,2)
        site_num = site.shape[0] #k
        centers = self.centers.reshape(self.__centers_num,1,2) #(m,1,2)
        dist = np.linalg.norm(centers - site,axis=2)#(m,k)
        dist = dist.reshape(self.__centers_num,site_num) #(m,k) 每个监测点距每个中心点的距离
        reduction = self.__model.propagation(dist,methods=method)#(m,k)
        point_power = self.center_power-reduction#(m,1)-(m,k)-->(m,k)
        return np.sum(point_power,axis=0,keepdims=True)

def generate_data(sample_num = 30):
    '''生成随机数据用于测试，数据不服从广播传播模型'''
    siteX = 100*np.random.rand(sample_num,1)#0-100
    siteY = 100*np.random.rand(sample_num,1)#0-100
    frequency = 40*np.random.rand(sample_num,1)+60 #60-100
    power = 70*np.random.rand(sample_num,1) #0-70
    input_data = np.array([siteX,siteY,frequency,power]).reshape(sample_num,4)
    return input_data

def main(data,test_site):
    '''
    data:       输入数据-->(latitude,longtitude,frequency,power)*N
    sample_num: 输入的坐标点数量
    test_site:  待求点的坐标，输出该点能量强度,支持多点输入
    '''
    model = broadcast_model(data)
    model.find_all_source(epochs=10,visualization=True)
    print('最终发射源坐标预测：',model.centers)
    power = model.predict_point_power(test_site) #获取测试点能量
    #print('中心点能量强度：',model.center_power)
    print('测试点能量强度：',power)

if __name__ == "__main__":
    dataset = generate_data()
    test_data = generate_data(sample_num=10)
    main(dataset,test_data[:,:2])
    #main() 参数待补充