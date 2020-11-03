import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

low = 0
high = 400000
x = np.arange(low, high, 1)

result_folder = '/home/gym/asml_experiment_result/'

files_to_be_draw = [result_folder+'result'+str(i)+'.json' for i in [1,11,12]]
color = ['r','y','g','c','b']
lable = ['batch = 8 lamda = 0.0001','batch = 1 lamda = 0.0001','update lamda = 0.0001']

###### heat map ######
file = files_to_be_draw[0]
with open(file,'r') as load_f:
    load_dict = json.load(load_f)
    for data in load_dict:
        if len(data['input_hidden']) > 0:
            x = np.array(data['input_hidden'])
            fig_name = 'input_hidden_'+str(data['id'])+'.png'
            fig_path = result_folder + fig_name
            fig = sns.heatmap(x, linewidths=.5,annot=True,vmax=1,vmin=-1)
            heatmap = fig.get_figure()
            heatmap.savefig(fig_path, dpi = 400,bbox_inches='tight')
            plt.close()
            x = np.array(data['hidden_output'])
            fig_name = 'hidden_output_'+str(data['id'])+'.png'
            fig_path = result_folder + fig_name
            fig = sns.heatmap(x, linewidths=.5,annot=True,vmax=1,vmin=-1)
            heatmap = fig.get_figure()
            heatmap.savefig(fig_path, dpi = 400,bbox_inches='tight')
            plt.close()


# ###### line ######
# for i in range(3):
#     file = files_to_be_draw[i]
#     c = color[i]
#     l = lable[i]
#     with open(file,'r') as load_f:
#         load_dict = json.load(load_f)
#         y=[]
#         for data in load_dict[low:high]:
#             y.append(data['loss'])
#         y = np.array(y)
#         plt.plot(x, y, color=c, linestyle="-", linewidth=1, label=l)
#     print("done {}".format(i))
#
# plt.legend(loc='upper left', bbox_to_anchor=(0.5, 0.95))
# plt.title("loss J")
# plt.show()
