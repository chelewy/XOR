import numpy as np
import matplotlib.pyplot as plt


x=np.array([[0,0,1,1],[0,1,0,1]])   # Giriş değerleri
y=np.array([[0,1,1,0]])             # Gerçek çıkış değerleri     
m = x.shape[1]
lr = 0.1                            # Learning rate
np.random.seed(2)                   # Başlangıç için random ağırlık değerleri
w1 = np.random.rand(2,2)            # 1. derece hidden layer ağırlıklarının bulunduğu matris
w2 = np.random.rand(1,2)            # 2. derece hidden layer ağırlıklarının bulunduğu matris
lossliste = []                      # kayıp listesi


# Sigmoid Fonksiyon
def sigmoid(z):
    z= 1/(1+np.exp(-z))
    return z

# Eğitim
def egtm(w1,w2,x):
    z1 = np.dot(w1,x)
    a1 = sigmoid(z1)    
    z2 = np.dot(w2,a1)
    a2 = sigmoid(z2)
    return z1,a1,z2,a2

# Backpropagation
def backpropagation(m,w1,w2,z1,a1,z2,a2,y):
    
    dz2 = a2-y
    dw2 = np.dot(dz2,a1.T)/m
    dz1 = np.dot(w2.T,dz2) * a1*(1-a1)
    dw1 = np.dot(dz1,x.T)/m
    return dw2,dw1

w11list=[]
w12list=[]
w22list=[]
w21list=[]
w3list=[]
w4list=[]
tekrar = 100000
for i in range(tekrar):
    z1,a1,z2,a2 = egtm(w1,w2,x)
    loss = -(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    lossliste.append(loss)
    dw2,dw1 = backpropagation(m,w1,w2,z1,a1,z2,a2,y)
    w2 = w2-lr*dw2
    w1 = w1-lr*dw1
    w11list.append(w1[0][0])
    w12list.append(w1[0][1])
    w21list.append(w1[1][0])
    w22list.append(w1[1][1])
    w3list.append(w2[0][0])
    w4list.append(w2[0][1])
    



plt.figure(figsize=(20,10))
plt.subplot(2,2,1)   
plt.plot(lossliste)
plt.xlabel("Eğitim Tekrarı")
plt.ylabel("Kayıp Değerleri")


plt.subplot(2,2,2)   
plt.plot(w11list)
plt.plot(w12list)
plt.plot(w21list)
plt.plot(w22list)
plt.plot(w3list)
plt.plot(w4list)
plt.xlabel("Eğitim Tekrarı")
plt.ylabel("Ağırlık Değişimleri")



def testfonk(w1,w2,giris):
    z1,a1,z2,a2 = egtm(w1,w2,test)
    print("x1=",test[0]," x2=",test[1]," : y=",a2[0])

test = np.array([[1],[0]])
testfonk(w1,w2,test)
test = np.array([[0],[0]])
testfonk(w1,w2,test)
test = np.array([[0],[1]])
testfonk(w1,w2,test)
test = np.array([[1],[1]])
testfonk(w1,w2,test)
