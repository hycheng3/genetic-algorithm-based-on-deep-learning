from skimage import measure
#from skimage import filters
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import math
from scipy.sparse.linalg import spsolve
from scipy import sparse
#import random
import scipy.io as sio
#import time

#import os
#os.remove("dens.csv")
#os.remove("Trans.csv")
#os.remove("Phase.csv")
np.random.seed(0)

def connect_analysis(blobs):
    gap = 3
    numc = np.size(blobs,0)
    m_label = np.zeros((numc,1))
    aera_label = np.zeros((numc,1))
    for i in range(numc):
        tem1 = blobs[i,:].reshape(20,20)
        x = np.ones((40, 40 * 4 + 2 * gap))
        x[0:20, gap : 20 + gap] = tem1
        x[39:19:-1, gap : 20 + gap] = tem1
        x[:, 20 + gap : 40 + gap] = x[:, 20 + gap - 1 : gap - 1 : -1]
        
        x[:, 40 + gap : 40 * 2 + gap] = x[:, gap : 40 + gap]           
        x[:, 40 * 2 + gap : 40 * 3 + gap] = x[:, gap : 40 + gap]
        x[:, 40 * 3 + gap : 40 * 4 + gap] = x[:, gap : 40 + gap]
        
        #all_labels = measure.label(blobs)
        blobs_labels = measure.label(x, background=0)
        m_label[i,0] = np.max(blobs_labels)
        
        counts = np.bincount(blobs_labels.reshape(40 * (40 * 4 + 2 * gap)).astype(int))
        flag = np.argmax(counts[1:])
        aera_label[i,0] = np.sum(((blobs_labels!=flag)+(blobs_labels!=0))==2)/(40 * (40 * 4 + 2 * gap))
     
    min_con_size = np.zeros((numc,1))    
    min_con_size_num = np.zeros((numc,1)) 
    for i in range(numc):
        flag1 = 10
        flag2 = 10

        tem1 = blobs[i,:].reshape(20,20)
        x = np.ones((40 + 2 * gap, 40 * 4 + 2 * gap))
        x[gap : 20 + gap, gap : 20 + gap] = tem1
        x[40 + gap - 1 : 20 + gap - 1 : -1, gap : 20 + gap] = tem1
        x[gap : 40 + gap, 20 + gap : 40 + gap] = x[gap : 40 + gap, 20 + gap - 1 : gap - 1 : -1]

        x[gap : 40 + gap, 40 + gap : 40 * 2 + gap] = x[gap : 40 + gap, gap : 40 + gap]   
        x[gap : 40 + gap, 40 * 2 + gap : 40 * 3 + gap] = x[gap : 40 + gap, gap : 40 + gap]
        x[gap : 40 + gap, 40 * 3 + gap : 40 * 4 + gap] = x[gap : 40 + gap, gap : 40 + gap]
        
        x[0 : gap : 1, :] = x[2 * gap - 1 : gap - 1 : -1, :]
        x[40 + gap : 40 + 2 * gap, :] = x[40 + gap - 1 : 39 : -1, :]       
        
        for iy in range(gap, 40 + gap):
            flag = gap
            ch = 0
            for ix in range(gap, 40 * 4 + 2 * gap):
               if x[iy, ix] == 1: 
                   ch = 1 
                   flag = flag + 1
               else: 
                   if flag > 0 and ch > 0:
                      flag1 = np.min((flag1, flag))
                      if flag < gap:
                         min_con_size_num[i,0] = min_con_size_num[i,0] + 1
                   ch = 0
                   flag = 0
            if flag > 0 and ch > 0:
               flag1 = np.min((flag1, flag))
               if flag < gap:
                  min_con_size_num[i,0] = min_con_size_num[i,0] + 1

            
        for ix in range(gap, 40 * 4 + gap):
            flag = 0
            for iy in range(0, 40 + 2 * gap):
               if x[iy,ix] == 1: 
                   flag = flag + 1
               else: 
                   if flag > 0 and iy >= gap:
                      flag2 = np.min((flag2, flag))
                      if flag < gap:
                         min_con_size_num[i,0] = min_con_size_num[i,0] + 1
                   flag = 0
            if flag > 0 and iy < 40 + gap:
               flag2 = np.min((flag2, flag)) 
               if flag < gap:
                  min_con_size_num[i,0] = min_con_size_num[i,0] + 1                    
        min_con_size[i,0] = np.min((flag1, flag2))       
    min_con_size_num = min_con_size_num/4      
    return m_label, min_con_size, min_con_size_num, aera_label

def reparing(blobs):
    num0 = np.size(blobs,0)
    den_new = np.zeros((num0, 20*20))
    tem0 = np.ones((22,22))
    tem1 = np.zeros((20,20))
    for numi in range(num0):
        tem0[1:21,1:21] = blobs[numi,:].reshape(20,20)
        tem0[0,:] = tem0[1,:]
        tem0[21,:] = tem0[20,:]
        tem0[:,0] = tem0[:,1]
        tem0[:,21] = tem0[:,20]
        for i in range(1,21):
            for j in range(1,21):
                if (tem0[i,j] + tem0[i-1,j] + tem0[i,j-1] + tem0[i+1,j] + tem0[i,j+1])/5 >= 0.5:
                    tem1[i-1, j-1] = 1
                    #tem0[i,j]=1
                else:
                    tem1[i-1, j-1] = 0
                    #tem0[i,j]=0
        #[numi,:]=tem0[1:21,1:21].reshape(1,20*20)
        den_new[numi,:] = tem1.reshape(1,20*20)
    return den_new


import tensorflow as tf
input_dim = 20
output_dim = 3
ave = 0
sigma = 0.25

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, input_dim, input_dim, 1])  
y = tf.placeholder(tf.float32, [None, output_dim])
def my_conv2d(x, in_ch, out_ch, me, std, ker_sh1, ker_sh2, str1, str2, activation_L1, pad):
    kernel = tf.Variable(tf.random_normal(shape=[ker_sh1,ker_sh2,in_ch,out_ch],mean = me,stddev = std))
    b = tf.Variable(tf.zeros([out_ch]))
    tf.add_to_collection("p_var",kernel)
    tf.add_to_collection("p_var",b)
    if activation_L1 is None:
        L = tf.nn.conv2d(x, kernel, strides=[1,str1,str2,1], padding=pad) + b
    else:  
        L = activation_L1(tf.nn.conv2d(x, kernel, strides=[1,str1,str2,1], padding=pad) + b)
    return L

def add_layer(L_Prev, num_nodes_LPrev, num_nodes_LX, activation_LX):
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights_LX = tf.Variable(tf.random_normal([num_nodes_LPrev,num_nodes_LX],stddev=sigma),name = 'w')  
			tf.add_to_collection("p_var",Weights_LX)
		with tf.name_scope('biases'):
			biases_LX = tf.Variable(tf.zeros([1, num_nodes_LX]),name = 'b')
			tf.add_to_collection("p_var", biases_LX)  
		with tf.name_scope('xW_plus_b'):
			xW_plus_b_LX = tf.matmul(L_Prev, Weights_LX) + biases_LX
		if activation_LX is None:
			LX = xW_plus_b_LX
		else:
			LX = activation_LX(xW_plus_b_LX)
		return LX

def add_layer_with_res(L_Prev, num_nodes_LPrev, num_nodes_LX, activation_LX):
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights_LX = tf.Variable(tf.random_normal([num_nodes_LPrev,num_nodes_LX], stddev=sigma), name = 'w')  
			tf.add_to_collection("p_var", Weights_LX)
		with tf.name_scope('biases'):
			biases_LX = tf.Variable(tf.zeros([1, num_nodes_LX]), name = 'b')
			tf.add_to_collection("p_var", biases_LX)  
		with tf.name_scope('xW_plus_b'):
			xW_plus_b_LX = tf.matmul(L_Prev,Weights_LX) + biases_LX
		if activation_LX is None:
			LX = xW_plus_b_LX
		else:
			LX = tf.add(activation_LX(xW_plus_b_LX), xW_plus_b_LX)
		return LX

y1 = my_conv2d(x,1,15,ave,sigma,2,2,1,1,tf.nn.relu,"SAME")        #[1,20,20,1] --> [3,3,1,3] --> [1,20,20,6]
y2 = my_conv2d(y1,15,30,ave,sigma,3,3,2,2,tf.nn.relu,"SAME")       #[1,20,20,6] --> [3,3,6,18] --> [1,10,10,18]  
y3 = my_conv2d(y2,30,60,ave,sigma,3,3,2,2,tf.nn.relu,"SAME")       #[1,10,10,18] --> [3,3,18,36] --> [1,5,5,36]
y4 = my_conv2d(y3,60,120,ave,sigma,3,3,2,2,tf.nn.relu,"SAME")       #[1,5,5,36] --> [3,3,36,72] --> [1,3,3,72]
y5 = my_conv2d(y4,120,240,ave,sigma,3,3,2,2,tf.nn.relu,"SAME")       #[1,3,3,72] --> [3,3,72,72] --> [1,2,2,72]

y101 = tf.reshape(y5, [tf.shape(x)[0],2*2*240])
y102 = add_layer(y101, 960, 120, tf.nn.tanh)
y103 = add_layer(y102, 120, 15, tf.nn.tanh)
y104 = add_layer(y103, 15, output_dim, tf.nn.tanh)

print('y1,', y1.shape)
print('y2,', y2.shape)
print('y3,', y3.shape)
print('y4,', y4.shape)
print('y5,', y5.shape)

print('y101,', y101.shape)
print('y102,', y102.shape)
print('y103,', y103.shape)
print('y104,', y104.shape)

prediction1 = tf.reshape(y104[0:tf.shape(x)[0],:], [tf.shape(x)[0], output_dim])

sess_real = tf.Session()
sess_real.run(tf.global_variables_initializer())
saver_real = tf.train.Saver(tf.get_collection_ref("p_var"))
saver_real.restore(sess_real,'./real/savemodel/model')

def NN_based_real_calculation(blobs):
	NN_real = sess_real.run(prediction1, feed_dict={x:blobs})
	return NN_real

sess_virtual = tf.Session()
sess_virtual.run(tf.global_variables_initializer())
saver_virtual = tf.train.Saver(tf.get_collection_ref("p_var"))
saver_virtual.restore(sess_virtual,'./virtual/savemodel/model')

def NN_based_virtual_calculation(blobs):
	NN_virtual = sess_virtual.run(prediction1,feed_dict={x:blobs})
	return NN_virtual

def NN_based_trans_phase_calculation(blobs):
	d_num = np.size(blobs,0)
	dataset0 = np.zeros((d_num, input_dim, input_dim, 1))
	for i in range(d_num):
	    dataset0_flag = np.array(blobs[i]).reshape(input_dim, input_dim)
	    for j in range(input_dim):
	        for k in range(input_dim):
	            dataset0[i][j][k][0] = dataset0_flag[j][k]	
    x_data = (np.array(dataset0) - 0.5) * 2
	rl = NN_based_real_calculation(x_data)
	vr = NN_based_virtual_calculation(x_data)
	cp = rl + vr * 1j
	T_NN = np.abs(cp)
	P_NN = np.angle(cp)
	T_NN = (T_NN < 1.0) * T_NN + (T_NN >= 1.0) * 1.0
	P_NN = (P_NN >= 0) * P_NN + (P_NN < 0) * (P_NN + 2 * math.pi)
	return T_NN, P_NN 

            
def trans_phase_calculation(blobs):
    numc = np.size(blobs, 0)
    T = np.zeros((numc, 3))
    P = np.zeros((numc, 3))
    for i in range(numc):
        x = blobs[i,:]
        T[i,:], P[i,:] = trans_phase_calculation1(x)
    return T, P
    
def trans_phase_calculation1(x0):
    #FEM for calculation   
    T = np.zeros(3)
    P = np.zeros(3)     
    return T, P  
    


def ga_algorithm(d_m, num_N, vol, it_num):
    rate1 = 0.05
    num = np.size(d_m,0)    
    dim = np.size(d_m,1) 
    num_N_N = 2 * num_N  
    #flag1=np.random.randint(num,size=(num_N_N,2))
    flag1 = np.round((np.random.random(size=(num_N_N,2))**1.1)*num-0.5).astype(int)
    den_f0 = np.zeros((2 * num_N_N, dim))
    for i in range(num_N_N): 
        cross_rate = 0.7 + 0.2 * ((np.random.rand()) - 0.5) / 0.5
        cross_dim = round(cross_rate*dim) 
        flag0 = np.random.randint(2,size=(dim))
        flag = np.random.permutation(dim)  
        a = d_m[flag1[i,0],:] 
        b = d_m[flag1[i,1],:]
        a_N = np.array(a)
        b_N = np.array(b)
        j = 0
        while j<dim:
              if j <= cross_dim:
                  a_N[flag[j]] = a[flag[j]] * (1 - flag0[j]) + b[flag[j]] * flag0[j]
                  b_N[flag[j]] = a[flag[j]] * flag0[j] + b[flag[j]] * (1 - flag0[j])
              if np.random.rand() < rate1:
                 a_N[flag[j]] = 1 - a_N[flag[j]]
              if np.random.rand() < rate1:
                 b_N[flag[j]] = 1 - b_N[flag[j]]
              j = j + 1
        den_f0[i*2,:] = a_N
        den_f0[i*2+1,:] = b_N
    #den_f=reparing(den_f0)
    if it_num <= 5:
        r_num = 2
    else:
        r_num = 1
    for i in range(r_num):
        den_f = reparing(den_f0)
        den_f0 = np.array(den_f)
    con_f, con_size_f, con_size_f_num, con_size_f_aera = connect_analysis(den_f)   
    nbt1 = 0.1
    nbt2 = 0.05
    nbt3 = 0.0005
    nbt4 = 0.1
    lis = sorted(range(len(nbt1 * (con_f - 1)**2 + nbt2 * (con_size_f < 3)**2 + nbt3 * (con_size_f_num - 0)**2 + nbt4 * (con_size_f_aera)**2)), key=lambda k: con_f[k])
    den = np.array(den_f[lis[0:num_N],:])
    con = np.array(con_f[lis[0:num_N],:])
    con_size = np.array(con_size_f[lis[0:num_N],:])
    con_size_num = np.array(con_size_f_num[lis[0:num_N],:])
    con_size_aera = np.array(con_size_f_aera[lis[0:num_N],:])
    return den, con, con_size, con_size_num, con_size_aera


    
def select_algorithm(g_all,num):
    lis=sorted(range(len(g_all)), key=lambda k: g_all[k])
    return lis[0:num]
    
di = 20*20    
num = 200
num_N = 200
iter_num = 150
vol = 0.5
nelx_hf = 20
nely_hf = 20
vol_N = round(vol * nelx_hf * nely_hf)


ini_d0 = ((np.random.rand(num,di)) > (1-vol)).astype(int)
for i in range(10):
   ini_d = reparing(ini_d0)
   ini_d0 = np.array(ini_d)
######### T,P=trans_phase_calculation(ini_d)
T,P = NN_based_trans_phase_calculation(ini_d)
ini_T = T #transmission 
ini_P = P #phase_shift
ini_C,ini_C_size,ini_C_size_num,ini_C_aera = connect_analysis(ini_d) #connectivity
#volume

#ini_p=np.zeros((200,3)) #transmission, phase_shift, connectivity, #volume
#for i in range(num):
#    ini_p[i][0:2]=trans_phase_calculation(ini_d[i][:])
#    ini_p[i][2]=connect_analysis(ini_d[i][:])
Td0 = 1.0
Td1 = 1.0
Td2 = 1.0
w = 2 * math.pi * np.array([9.5e4,10e4,10.5e4])
Pd0 = 150/180*math.pi*w[0]/w[1]
Pd1 = 150/180*math.pi
Pd2 = 150/180*math.pi*w[2]/w[1]
if Pd2 > 2 * math.pi:
   Pd2 = Pd2 - 2 * math.pi
Cd0 = 1.0
nbt0 = 1.0
mw = 0.6
nbt1 = 1.0
nbt2 = 0.1*0.2
nbt3 = 0.05*0.2
nbt4 = 0.0005*0.2
nbt5 = 0.1*0.2
ini_grade = np.zeros((num,1))
flag00 = abs(ini_P[:,0] - Pd0)
flag01 = 2 * math.pi - flag00
flag0 = (flag00<=flag01) * flag00 + (flag00>flag01) * flag01
flag10 = abs(ini_P[:,1] - Pd1)
flag11 = 2 * math.pi - flag10
flag1 = (flag10<=flag11) * flag10 + (flag10>flag11) * flag11
flag20 = abs(ini_P[:,2] - Pd2)
flag21 = 2 * math.pi - flag20
flag2 = (flag20<=flag21) * flag20 + (flag20>flag21) * flag21
ini_grade[:,0] = nbt0 * (mw * (ini_T[:,0] - Td0)**2 + (ini_T[:,1] - Td1)**2 + mw * (ini_T[:,2] - Td2)**2)\
               + nbt1 * (mw * flag0**2 + flag1**2 + mw * flag2**2)\
               + nbt2 * (ini_C[:,0] - Cd0)**2 + nbt3 * (ini_C_size[:,0]<3)**2 + nbt4 * (ini_C_size_num[:,0]-0)**2 + nbt5 * (ini_C_aera[:,0])**2
ini_stop1 = np.zeros((num,1))
ini_stop1[:,0] = (ini_T[:,1] - Td1)**2 + flag1**2
ini_stop2 = np.zeros((num,1))
ini_stop2[:,0] = nbt0 * (mw * (ini_T[:,0] - Td0)**2 + (ini_T[:,1] - Td1)**2 + mw * (ini_T[:,2] - Td2)**2)\
               + nbt1 * (mw * flag0**2 + flag1**2 + mw * flag2**2)
ini_stop3 = np.zeros((num,1))
ini_stop3[:,0] = nbt2 * (ini_C[:,0] - Cd0)**2 + nbt3 * (ini_C_size[:,0]<3)**2 + nbt4 * (ini_C_size_num[:,0] - 0)**2 + nbt5 * (ini_C_aera[:,0])**2
d_m = ini_d
T_m = ini_T
P_m = ini_P
C_m = ini_C
Cs_m = ini_C_size
Csn_m = ini_C_size_num
Ca_m = ini_C_aera
g_m = ini_grade
Stop1_m = ini_stop1
Stop2_m = ini_stop2
Stop3_m = ini_stop3
fi1 = open('dens.csv','ab')
np.savetxt(fi1, d_m, fmt='%1d', delimiter=',')
fi1.close()
fi2 = open('Trans.csv','ab')
np.savetxt(fi2, T_m, fmt='%e', delimiter=',')
fi2.close()
fi3 = open('Phase.csv','ab')
np.savetxt(fi3, P_m, fmt='%e', delimiter=',')
fi3.close()

Stop1_N = np.zeros((num_N, 1))
Stop2_N = np.zeros((num_N, 1))
Stop3_N = np.zeros((num_N, 1))
g_N = np.zeros((num_N, 1))
for i in range(iter_num):
    print(i)
    nbt0 = np.minimum(np.maximum(1, (i-5)/20+1), 5)
    #nbt1=np.minimum(np.maximum(0.25,0.25*(i/5+1)),10)
    nbt1 = np.minimum(np.maximum(1, (i-5)/20+1), 5)
    nbt2 = 0.1 * np.minimum(np.maximum(0.2, 0.2 * ((i-6)/6+1)), 1)
    nbt3 = 0.05 * np.minimum(np.maximum(0.2, 0.2 * ((i-6)/6+1)), 1)
    nbt4 = 0.0005 * np.minimum(np.maximum(0.2, 0.2 * ((i-6)/6+1)), 1)  
    nbt5 = 0.1 * np.minimum(np.maximum(0.2, 0.2 * ((i-6)/6+1)), 1)
    flag00 = abs(P_m[:,0] - Pd0)
    flag01 = 2 * math.pi - flag00
    flag0 = (flag00<=flag01) * flag00 + (flag00>flag01) * flag01
    flag10 = abs(P_m[:,1] - Pd1)
    flag11 = 2 * math.pi - flag10
    flag1 = (flag10<=flag11) * flag10 + (flag10>flag11) * flag11
    flag20 = abs(P_m[:,2] - Pd2)
    flag21 = 2 * math.pi - flag20
    flag2 = (flag20<=flag21) * flag20 + (flag20>flag21) * flag21
    g_m[:,0 ] = nbt0 * (mw * (T_m[:,0] - Td0)**2 + (T_m[:,1] - Td1)**2 + mw * (T_m[:,2] - Td2)**2)\
              + nbt1 * (mw * flag0**2 + flag1**2 + mw * flag2**2)\
              + nbt2 * (C_m[:,0] - Cd0)**2 + nbt3 * (Cs_m[:,0]<3)**2 + nbt4 * (Csn_m[:,0] - 0)**2 + nbt5 * (Ca_m[:,0])**2    
    Stop1_m[:,0] = (T_m[:,1] - Td1)**2 + flag1**2
    Stop2_m[:,0] = nbt0 * (mw * (T_m[:,0] - Td0)**2 + (T_m[:,1] - Td1)**2 + mw * (T_m[:,2] - Td2)**2)\
                 + nbt1 * (mw * flag0**2 + flag1**2 + mw * flag2**2)
    Stop3_m[:,0] = nbt2 * (C_m[:,0] - Cd0)**2 + nbt3 * (Cs_m[:,0]<3)**2 + nbt4 * (Csn_m[:,0] - 0)**2 + nbt5 * (Ca_m[:,0])**2  
    d_N, C_N, Cs_N, Csn_N, Ca_N = ga_algorithm(d_m, num_N, vol_N, i)
    #t1=time.time()

    T_N, P_N = NN_based_trans_phase_calculation(d_N)
    #print(time.time()-t1)
    fi1 = open('dens.csv','ab')
    np.savetxt(fi1, d_N, fmt='%1d', delimiter=',')
    fi1.close()
    fi2 = open('Trans.csv','ab')
    np.savetxt(fi2, T_N, fmt='%e', delimiter=',')
    fi2.close()
    fi3 = open('Phase.csv','ab')
    np.savetxt(fi3, P_N, fmt='%e', delimiter=',')
    fi3.close()
    '''
    if  i%10==0: 
        T_N_c,P_N_c=trans_phase_calculation(d_N)
        fi4=open('Trans_C.csv','ab')
        np.savetxt(fi4,T_N_c,fmt='%e',delimiter=',')
        fi4.close()
        fi5=open('Phase_C.csv','ab')
        np.savetxt(fi5,P_N_c,fmt='%e',delimiter=',')
        fi5.close()        
    '''
    #C_N=connect_analysis(d_N)
    #g_N=np.zeros((num_N,1))
    flag00 = abs(P_N[:,0] - Pd0)
    flag01 = 2 * math.pi - flag00
    flag0 = (flag00<=flag01) * flag00 + (flag00>flag01) * flag01
    flag10 = abs(P_N[:,1] - Pd1)
    flag11 = 2 * math.pi - flag10
    flag1 = (flag10<=flag11) * flag10 + (flag10>flag11) * flag11
    flag20 = abs(P_N[:,2] - Pd2)
    flag21 = 2 * math.pi - flag20
    flag2 = (flag20<=flag21) * flag20 + (flag20>flag21) * flag21
    
    g_N[:,0 ] = nbt0 * (mw * (T_N[:,0] - Td0)**2 + (T_N[:,1] - Td1)**2 + mw * (T_N[:,2] - Td2)**2)\
              + nbt1 * (mw * flag0**2 + flag1**2 + mw * flag2**2)\
              + nbt2 * (C_N[:,0] - Cd0)**2 + nbt3 * (Cs_N[:,0]<3)**2 + nbt4 * (Csn_N[:,0] - 0)**2 + nbt5 * (Ca_N[:,0])**2
    #Stop1_N=np.zeros((num_N,1))
    Stop1_N[:,0] = (T_N[:,1] - Td1)**2 + flag1**2
    #Stop2_N=np.zeros((num_N,1))
    Stop2_N[:,0] = nbt0 * (mw * (T_N[:,0] - Td0)**2 + (T_N[:,1] - Td1)**2 + mw * (T_N[:,2] - Td2)**2)\
                 + nbt1 * (mw * flag0**2 + flag1**2 + mw * flag2**2)
    #Stop3_N=np.zeros((num_N,1))
    Stop3_N[:,0] = nbt2 * (C_N[:,0] - Cd0)**2 + nbt3 * (Cs_N[:,0]<3)**2 + nbt4 * (Csn_N[:,0] - 0)**2 + nbt5 * (Ca_N[:,0])**2
    d_all = np.concatenate((d_m,d_N),axis=0)
    T_all = np.concatenate((T_m,T_N),axis=0)
    P_all = np.concatenate((P_m,P_N),axis=0)
    C_all = np.concatenate((C_m,C_N),axis=0)
    Cs_all = np.concatenate((Cs_m,Cs_N),axis=0)
    Csn_all = np.concatenate((Csn_m,Csn_N),axis=0)
    Ca_all = np.concatenate((Ca_m,Ca_N),axis=0)
    g_all = np.concatenate((g_m,g_N),axis=0)
    Stop1_all = np.concatenate((Stop1_m,Stop1_N),axis=0)
    Stop2_all = np.concatenate((Stop2_m,Stop2_N),axis=0)
    Stop3_all = np.concatenate((Stop3_m,Stop3_N),axis=0)
    lis = select_algorithm(g_all,num)
    d_m = np.array(d_all[lis,:])
    T_m = np.array(T_all[lis,:])
    P_m = np.array(P_all[lis,:])
    C_m = np.array(C_all[lis,:])
    Cs_m = np.array(Cs_all[lis,:])
    Csn_m = np.array(Csn_all[lis,:])
    Ca_m = np.array(Ca_all[lis,:])
    g_m = np.array(g_all[lis,:])
    Stop1_m = np.array(Stop1_all[lis,:])
    Stop2_m = np.array(Stop2_all[lis,:])
    Stop3_m = np.array(Stop3_all[lis,:])
    print(np.mean(T_m))
    print(np.mean(P_m))
    print(T_m[0,:])
    print(P_m[0,:])
    plt.figure(1)
    a = np.zeros((40,40))
    a[0:20, 0:20] = d_m[0,:].reshape(20,20)
    a[20:40, 0:20] = a[19::-1, 0:20]    
    a[0:40, 20:40] = a[0:40, 19::-1]
    plt.imshow(1-a, interpolation='None', cmap='gray') 
    plt.savefig('first.png')
 
    #plt.show()
    plt.figure(2)
    a[0:20, 0:20] = d_m[1,:].reshape(20,20)
    a[20:40, 0:20] = a[19::-1,0:20]    
    a[0:40, 20:40] = a[0:40, 19::-1]
    plt.imshow(1-a, interpolation='None', cmap='gray')
    plt.savefig('second.png')
    #plt.show()
    print(np.min(Stop1_m))
    print(Stop2_m[np.argmin(Stop1_m), 0])
    print(Stop3_m[np.argmin(Stop1_m), 0])
    print(g_m[np.argmin(Stop1_m), 0] / nbt0)
    index = np.where(Stop1_m < 0.01)
    if i>30 and np.sum(Stop1_m<0.015)>0 and np.sum(Stop3_m[index]/nbt3<1.2)>0 and np.sum(Stop2_m[index]/nbt0<0.07)>0: #and g_m[np.argmin(Stop1_m)]/nbt0<0.05:
        #if  i%10!=0: 
        #T_N_c,P_N_c=trans_phase_calculation(d_N)
        #fi4=open('Trans_C.csv','ab')
        #np.savetxt(fi4,T_N_c,fmt='%e',delimiter=',')
        #fi4.close()
        #fi5=open('Phase_C.csv','ab')
        #np.savetxt(fi5,P_N_c,fmt='%e',delimiter=',')
        #fi5.close()  
        break
                       
    
plt.figure(1)
a = np.zeros((40,40))
a[0:20, 0:20] = d_m[0,:].reshape(20,20)
a[20:40, 0:20] = a[19::-1, 0:20]    
a[0:40, 20:40] = a[0:40, 19::-1]
plt.imshow(1-a, interpolation='None', cmap='gray')
plt.savefig('first.png')

#plt.show()
plt.figure(2)
a[0:20, 0:20] = d_m[1,:].reshape(20,20)
a[20:40, 0:20] = a[19::-1, 0:20]    
a[0:40, 20:40] = a[0:40, 19::-1]
plt.imshow(1-a, interpolation='None', cmap='gray')
plt.savefig('second.png')
#plt.show()
