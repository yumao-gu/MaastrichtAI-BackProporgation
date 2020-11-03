import numpy as np
import json

###### parameters ######
alpha = 0.01
# epsilon = 0.0001
lamda = 0.0001
Lamda_2 = np.diag(np.array([0,lamda,lamda,lamda]))
Lamda_1 = np.diag(np.array([0,lamda,lamda,lamda,lamda,lamda,lamda,lamda,lamda]))
n_data = 8
n_nodes_input = 8
n_nodes_hidden = 3
n_nodes_output = 8

###### activation function ######
def sigmoid(x):
    return 1/(1 + np.exp(-x))

###### loss function ######
def J(W1,W2,h,y):
    return 1/2*(np.sum((h-y)**2)/n_data + lamda*np.sum(W1**2) + lamda*np.sum(W2**2))

###### input data ######
input_layer = \
[[0,0,0,0,0,0,0,1],\
 [0,0,0,0,0,0,1,0],\
 [0,0,0,0,0,1,0,0],\
 [0,0,0,0,1,0,0,0],\
 [0,0,0,1,0,0,0,0],\
 [0,0,1,0,0,0,0,0],\
 [0,1,0,0,0,0,0,0],\
 [1,0,0,0,0,0,0,0]]
input_layer = np.array(input_layer)
bias = np.array([[1,1,1,1,1,1,1,1]])
# bias = np.array([[1]])
input_layer = np.c_[bias.T,input_layer].T

###### weights initialization ######
input_hidden = np.random.normal(0, 0.01, (n_nodes_hidden,n_nodes_input + 1))  # 3x9 Matrix
hidden_output = np.random.normal(0, 0.01, (n_nodes_output,n_nodes_hidden + 1))  # 8x4 matrix

###### training ######
file = open('./result.json','w',encoding='utf-8')
all_data = []
###### iterations ######
for i in range(400000):
    gradient_J_2,gradient_J_1 = 0,0
    ###### different data ######
    for j in range(8):
        ###### forward ######
        a_1 = np.array([input_layer[:,j]]).T
        z_2 = np.dot(input_hidden,a_1)
        a_2 = sigmoid(z_2)
        a_2 = np.r_[np.array([[1]]),a_2]
        z_3 = np.dot(hidden_output,a_2)
        a_3 = sigmoid(z_3)

        ###### back propogation ######
        y = np.array([input_layer[1:,j]]).T
        delta_3 = -(y - a_3)*a_3*(1-a_3)
        delta_2 = hidden_output.T.dot(delta_3)*a_2*(1-a_2)
        delta_2 = delta_2[1:,:]

        gradient_J_2 += delta_3.dot(a_2.T)
        gradient_J_1 += delta_2.dot(a_1.T)
    
    ###### weights update ######
    hidden_output = hidden_output - alpha * (gradient_J_2/n_data + Lamda_2.dot(hidden_output.T).T)
    input_hidden = input_hidden - alpha * (gradient_J_1/n_data + Lamda_1.dot(input_hidden.T).T)

    ###### loss each iteration ######
    a_1 = input_layer
    z_2 = np.dot(input_hidden,a_1)
    a_2 = sigmoid(z_2)
    a_2 = np.r_[bias,a_2]
    z_3 = np.dot(hidden_output,a_2)
    a_3 = sigmoid(z_3)
    loss = J(input_hidden[:,1:],hidden_output[:,1:],a_3,a_1[1:,:])

    ###### experiment recording ######
    data = {}
    data['id'] = i
    data['loss'] = loss
    if i%50000 == 0:
      data["hidden_output"] = hidden_output.tolist()
      data["input_hidden"] = input_hidden.tolist()
    else:
      data["hidden_output"] = []
      data["input_hidden"] = []
    all_data.append(data)
    print(loss)

json.dump(all_data,file,ensure_ascii=False)
file.close()


print(a_1, "\n ", np.where(a_3 > 0.5 , 1, 0))
print(a_3)
