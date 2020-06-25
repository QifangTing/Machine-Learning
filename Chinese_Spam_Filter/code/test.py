import os
import train
import re
import jieba

p_s = 0.5		# 垃圾/健康邮件的先验概率
p_h = 0.5
condition_xw = []
condition_wx = []
word_to_id_map = {}

'''
	函数get_condition_xw：计算词汇表中所有单词的条件概率P(s|w) P(h|w)
'''
def get_condition_xw(condition_wx):
	for i in range(0, len(condition_wx)):
		p_ws = condition_wx[i][1]
		p_wh = condition_wx[i][0]
		p_sw = (p_ws * p_s) / (p_ws * p_s + p_wh * p_h)
		p_hw = (p_wh * p_h) / (p_wh * p_h + p_ws * p_s)
		condition_xw.append([p_hw, p_sw])

'''
	函数get_condition_xW：计算一篇邮件中的P(s|W)和P(h|W)
'''
def get_condition_xW(file):
	with open(file, 'r', encoding='gbk') as reader:
		rule = re.compile(r"[^\u4e00-\u9fa5]")				# 过滤掉非中文字符
		line = reader.read()
		content = rule.sub('', line)
		initial_words = jieba.lcut(content)  				# jieba分词获得初始文字列表
		processed_words = train.move_stops(initial_words)  	# 删除停用词
	
	words = list(set(processed_words))     					# 去掉列表中的重复值
	p_s_W = 1        # P(s | w_1, w_2, .......,w_n)
	p_W_s = 1        # P(w_1, w_2,......, w_n | s)
	p_W_h = 1		 # P(w_1, w_2,......, w_n | h)
	
	for i in range(0, len(words)):
		if words[i] in word_to_id_map.keys():		# id的转换
			id = word_to_id_map[words[i]]
			p_W_s *= condition_wx[id][1]  # 朴素贝叶斯假设变量的各特征是相互独立的
			p_W_h *= condition_wx[id][0] 
		else:
			p_W_s *= 0.4         # 如果一个单词没出现过, 无法获取P(w|s), 假定其等于0.4,
			p_W_h *= 0.6         # 因垃圾邮件往往是固定的词语, 如果单词从没出现过多半是正常的

	p_s_W = (p_W_s * p_s)        # 因分母一样, 故只考虑分子最大化
	p_h_W = (p_W_h * p_h)
	return p_s_W, p_h_W

'''
	函数read_file: 读取全部文件,进行条件概率P(s|W)的运算
'''
def read_file(path):
	files = os.listdir(path)
	category = {}               # 记录分类
	for name in files:
		file = os.path.join(path, name)
		(p_s_W, p_h_W) = get_condition_xW(file)
		if p_s_W > p_h_W:
			category[name] = 1      # 1表示垃圾邮件
		else:
			category[name] = 0      # 0表示健康邮件
	return category

'''
	函数show_result: 计算并输出准确率
'''
def show_result(category):
	accuracy = 1               
	correct_quantity = 0        # 被正确分类的数目
	for name in category.keys():
		if int(name) >= 200:        # 实际是垃圾邮件
			if category[name] == 1:
				correct_quantity += 1
		else:                       # 实际是健康邮件
			if category[name] == 0:
				correct_quantity += 1
	accuracy = correct_quantity / len(category.keys())
	print("The accuracy is " + str(accuracy*100) + '%')

'''
	函数read_model:从持久化的文件中读取模型的参数, 填充给相应变量
'''
def read_model():
	file_vocabulary = '../data/word_probability.txt'
	file_id_map = '../data/word_set.txt'
	with open(file_vocabulary, 'r') as reader:
		for line in reader.readlines():
			list = line.strip().split('\t')
			if len(list) == 2:
				condition_wx.append([float(list[0]), float(list[1])])
	with open(file_id_map, 'r') as reader:
		for line in reader.readlines():
			list = line.strip().split('\t')
			if len(list) == 2:
				word_to_id_map[list[0]] = int(list[1])

if __name__ == '__main__':
	read_model()
	get_condition_xw(condition_wx)	# 转换词汇表中全部
	category = read_file('../data/test')
	show_result(category)
