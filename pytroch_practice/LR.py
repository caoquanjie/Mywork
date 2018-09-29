import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
fig1=plt.figure()
ax1 = fig1.add_subplot(111)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.mlab as mlab

#outputfile = 'C:\\Users\\caodada\\Desktop\\test_data\\'
f1 = open('betweenness.txt','r')
f2 = open('degree.txt','r')
f3 = open('networkx structure scale.txt','r')


x = []
y = []
z = []
for readline in f1.readlines():
      x.append(eval((readline).strip()))
x = np.array([x]).reshape(118,1)
#print(x)

for readline in f2.readlines():
    y.append(eval((readline).strip()))
y = np.array([y]).reshape(118,1)

for readline in f3.readlines():
    z.append(eval((readline).strip()))

z = np.array([z]).reshape(118,1)

data = np.concatenate((x,y,z),1)
#print(data)

data = np.random.permutation(data)
train_data = data[:82]

#train_data = train_data[:,:2]
test_data = data[82:]


train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = train_data[:,2].unsqueeze(1)


test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = test_data[:,2].unsqueeze(1)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=20, n_output=1)     # define the network
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.MSELoss()


for t in range(200):
    prediction = net(train_data[:,:2])
    loss = loss_func(prediction, train_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        print('the total loss is%.4f'%loss)

'''
#画训练集三维图
x_list= np.zeros((82,82,82))

p_list= np.zeros((82,82,82))


x=list(train_data[:,0].data.numpy())
print(type(x))
y=list(train_data[:,1].data.numpy())
z=list(train_data[:,2].data.numpy())

p=list(prediction.data.numpy())

for i in range(0,82):
    x_list[i,0]=x[i]
    x_list[i,1]=y[i]
    x_list[i,2]=z[i]

    p_list[i,0]=x[i]      
    p_list[i,1]=y[i]
    p_list[i,2]=p[i]

ax = fig1.gca(projection='3d') # 得到3d坐标的图

for x in x_list:
    ax.scatter(x[0],x[1],x[2],c='blue')
    #plt.savefig("d3_image.png")
    
for x in p_list:
    ax.scatter(x[0],x[1],x[2],c='red')
    #plt.savefig("d3_image.png")

ax.set_xlabel("betweenness", color='black')
ax.set_ylabel("degree", color='black')
ax.set_zlabel("networkx structure scale", color='black')

plt.show()
'''

'''
#画测试集三维图
x_list= np.zeros((36,36,36))

p_list= np.zeros((82,82,82))


x=list(test_data[:,0].data.numpy())
print(type(x))
y=list(test_data[:,1].data.numpy())
z=list(test_data[:,2].data.numpy())

test=net(train_data[:,:2])
p=list(test.data.numpy())

for i in range(0,36):
    x_list[i,0]=x[i]
    x_list[i,1]=y[i]
    x_list[i,2]=z[i]

    p_list[i,0]=x[i]      
    p_list[i,1]=y[i]
    p_list[i,2]=p[i]

ax = fig1.gca(projection='3d') # 得到3d坐标的图

for x in x_list:
    ax.scatter(x[0],x[1],x[2],c='blue')
    #plt.savefig("d3_image.png")
    
for x in p_list:
    ax.scatter(x[0],x[1],x[2],c='red')
    #plt.savefig("d3_image.png")

ax.set_xlabel("betweenness", color='black')
ax.set_ylabel("degree", color='black')
ax.set_zlabel("networkx structure scale", color='black')

plt.show()
'''


#绘制平面图
fig = plt.figure()  # 得到画面
ax = fig.gca(projection='3d')  # 得到3d坐标的图

x = np.arange(0,0.25,0.01)
y = np.arange(0,8,0.1)

Z=np.zeros((80,25))

tmp = []
for i in list(x):
    for j in list(y):
        tmp.append([i,j])
test_data = torch.Tensor(tmp)
predict_test=net(test_data)
print(predict_test)
predict_test=predict_test.data.numpy()
print(np.shape(predict_test))

predict_test=np.squeeze(predict_test)
print(predict_test)

ii=0
for i in range(25):
    for j in range(80):
        Z[j,i]=predict_test[ii]
        ii+=1

print(type(Z))

X,Y = np.meshgrid(x, y)
print(X)
print(Y)
   # 曲面，x,y,z坐标，横向步长，纵向步长，颜色，线宽，是否渐变
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(0.01, 1.01)

ax.set_xlabel("betweenness", color='black')
ax.set_ylabel("degree", color='black')
ax.set_zlabel("networkx structure scale", color='black')

#ax.zaxis.set_major_locator(LinearLocator(10))  # 设置z轴标度
#ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))  # 设置z轴精度
    # shrink颜色条伸缩比例0-1, aspect颜色条宽度（反比例，数值越大宽度越窄）
#fig1.colorbar(surf, shrink=0.5, aspect=5)

#plt.savefig("d3_hookface.png")
plt.show()













#print(data_test)

'''
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

prediction = net(train_data[:,:2])


# 曲面，x,y,z坐标，横向步长，纵向步长，颜色，线宽，是否渐变
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(0, 1.01)

'''
'''
ax.set_xlabel("x-label", color='r')
ax.set_ylabel("y-label", color='g')
ax.set_zlabel("z-label", color='b')

ax.zaxis.set_major_locator(LinearLocator(10))  # 设置z轴标度
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))  # 设置z轴精度
    # shrink颜色条伸缩比例0-1, aspect颜色条宽度（反比例，数值越大宽度越窄）
fig1.colorbar(surf, shrink=0.5, aspect=5)
'''
'''
plt.savefig("d3_hookface.png")
plt.show()

'''


















