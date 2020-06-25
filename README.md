# Machine-Learning  
## 第一次机器学习实验：用KNN算法实现MNIST的数字识别  

    人工智能导论作业是大一下做的CNN实现MNIST手写数字的识别，在这里是重温与对比理解KNN算法。
    第一次作业的文档报告，介绍了本次作业的大致思路和结果记录。
    测试集和训练集是对MNIST中的图像已经处理过的二进制文件。

## 第二次机器学习实验：Naive Bayes算法实现垃圾邮件分类
    
    通过学习老师分享的英文垃圾邮件分类（Naive Bayes实现，链接如下），对朴素贝叶斯分类器在垃圾邮件的分类上有了理解。
    （https://blog.csdn.net/asialee_bird/article/details/81288955?tdsourcetag=s_pctim_aiomsg）
    中文邮件分类的数据集大很多，中文的词汇也较英语复杂变化，引入库jieba（结巴分词）和中文停用词。
    
    优化使用了拉普拉斯平滑。开始简单粗暴使用0.01替代为0的概率，准确率在95+。
    观察输出的数据概率文件word_probability.txt，发现因为0.01与其他出现较少的频率相差过大。
    可以采用0.0001代替也可提升准确度在97+。用拉普拉斯平滑，默认最少出现一次得到的最后的准确度在98.21%
    
    文件夹下data包含ham、spam、test的数据集、stop_chinese和已经训练生成的word_set,word_probability；
           code中有train.py和test.py两个源代码文件； record包含一些截图。

## 结课项目： 基于朴素贝叶斯分类器的语音性别识别

