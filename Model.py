import numpy as np

class model:
    '''传播模型'''
    def __init__(self,launch_frequency = 100,launch_height = 5,receive_height = 30):
        self.f = launch_frequency #MHz
        self.h_te = launch_height
        self.h_re = receive_height

    def __get_alpha(self, f,h_re,city_type):
        '''city_type："big" "middle_and_small" '''
        if city_type == 'big': #大城市
            if f<300:
                return 8.29*(np.log10(1.54*h_re)**2)-1.1 #dB
            else:
                return 3.2*(np.log10(11.75*h_re)**2)-4.97 #dB

        else:  #中小城市
            return h_re*(1.11*np.log10(f)-0.7)-(1.56*np.log10(f)-0.8); #dB
        
    def free_propagation(self,d,f = None):
        '''自由传播模型'''
        if f == None:
            f = self.f
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
        alpha = self.__get_alpha(f,self.h_re,"middle_and_small")
        power_loss =69.55+26.16*np.log10(f)-13.82*np.log10(self.h_te) - alpha + (44.9-6.55*np.log10(self.h_te))*np.log10(d+1e-6)
        return np.maximum(power_loss,0)

    def Egli(self,d, f = None):
        '''
        Egli传播模型
        适用传输距离:小于60km;
        通用频率：40 -450MHz (小于1000MHz时也可以使用)，对于预测丘陵地形场强较为准确。
        '''
        if f == None:
            f = self.f
        return 88+40*np.log10(d)- 20*np.log10(self.h_te*self.h_re)+20*np.log10(f)

    def propagation(self,d, f = None,methods = 'Okumura_Hata'):
        '''选择使用的模型
            methods：可选'free' 或 'Okumura_Hata' 或 'Egli'  '''
        if methods == 'free':
            return self.free_propagation(d,f)
        elif methods == 'Okumura_Hata':
            return self.Okumura_Hata(d,f)
        elif methods == 'Egli':
            return self.Egli(d,f)
