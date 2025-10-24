from numpy import*
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group=array([[1.0,1,1],[1.0,1.0],[0,0],[0,0.1]])
    labels=["A","A","B","B"]
    return group,labels

def file2matrix(filename):
    fr=open(filename)   #打开文件
    array_olines=fr.readlines() #从文件中读取一行
    number_lines=len(array_olines) 
    return_mat=zeros((number_lines,3))
    class_label_vector=[]
    index=0
    for line in array_olines:
        line=line.strip()
        list_from_line=line.split('\t')
        return_mat[index,:]=list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index+=1
    return return_mat,class_label_vector


file_path = r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt"
dating_data_mat, dating_labels = file2matrix(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

fig=plt.figure()
#图形中添加一个子图
#第一个数字 "1"：将图形划分为 1 行
#第二个数字 "1"：将图形划分为 1 列
#第三个数字 "1"：选择第 1 个（也是唯一的一个）子图位置
ax=fig.add_subplot(111)  

#scatter 绘制散点图
#选择第二列作为图的x轴，选择第三列作为y轴
ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2])
#第三个参数调整点的大小，第四个参数控制点的颜色
ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2],15.0*array(dating_labels),15.0*array(dating_labels))
ax.set_xlabel('第二特征')
ax.set_ylabel('第三特征')
ax.set_title('特征关系散点图')
#plt.show()


def autoNorm(dataSet):
    #数据进行归一化

    minVals = dataSet.min(0) # 计算每列的最小值
    maxVals = dataSet.max(0)  # 计算每列的最大值
    ranges = maxVals - minVals  # 计算每列数据的范围（最大值-最小值）
    normDataSet = zeros(shape(dataSet))    # 创建一个与原始数据集形状相同的零矩阵，用于存储归一化后的数据
    m = dataSet.shape[0]     # 获取数据集的行数
    normDataSet = dataSet - tile(minVals, (m, 1))   # 将原始数据减去最小值（使用tile函数将minVals复制m行，使其形状与dataSet匹配）
    normDataSet = normDataSet/tile(ranges, (m, 1))     # 将减去最小值后的数据除以范围，实现归一化到[0,1]范围
    return normDataSet, ranges, minVals   # 返回归一化后的数据集、每列的范围和每列的最小值

def datingClassTest():
    # 使用留出法：设置测试集比例（hold-out比例），这里使用50%的数据作为测试集
    hoRtio = 0.50
    datingDataMat,datinglabels = file2matrix(file_path)
    normMat,ranges,minvals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRtio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :]  norMat[numTestVecs:m, :], datingLabels[numTestvecs:m]. 3)
        print(f'分类器的预测结果: {classifierResult},真实结果:  {datinglabels[i]}')

#datingClassTest()

def classify0(inX, dataSet, labels, k):
   #KNN分类核心代码

def classify_person():
    """
    交互式输入三项特征，使用约会数据集做 KNN 分类，并输出印象结果。
    """
    result_list = ['不感兴趣'，'有点兴趣','非常有兴趣']
               
    try:
        percent_tats = input("业余时间花费在视频游戏上的时间比率 (0~1),输入q退出 : ")
        if percent_tats.lower() in ['q','exit']:
            return None 
        percent_tats = float(percent_tats)
        ff_miles = float(input("每年飞行公里数:"))
        ice_cream = float(input("每年消耗冰淇淋的升数:"))
    except ValueError:
        print("输入必须是数字，请重新尝试。")
        return Ture 

    datingDataMat,datinglabels = file2matrix(file_path)
    normMat,ranges,minvals = autoNorm(datingDataMat)
    in_arr = array([ff_miles,percent_tats,ice_cream])
    classifier_result = classify0((in_arr_minVals)/ranges,normMat,datinglabels,3)

    idx = int(classifier_result) -1


    desc = result_list[idx] if 0 <= idx < len(result_list)else str(classifier_result)
    print(f"你对这个人的印象是:{desc}")
    return True


 # —— 死循环调用 ——
if __name__ == "__main__": 
    print("=== 约会数据 KNM 测试数据 ===")
    print("输入 'q' 或 'exit' 可以退出。")
    while True:
        flag = classify_person()
        if flag is None:
            print("程序已退出。")
            break