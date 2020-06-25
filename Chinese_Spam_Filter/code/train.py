import jieba
import re
import os

vocabulary = []     			# 二维列表，记录每个单词及其在健康/垃圾邮件出现次数
word_to_id_map = {}				# 字典,记录单词和其出现的顺序
total_mail = [0, 0]             # q[0/1]是健康/垃圾邮件的数量

'''
	函数get_stops： 将停用词存储在列表stop_word中
'''
def get_stops():
	stop_words = []		
	with open('../data/stop_chinese.txt', 'r', encoding='gbk') as reader:
		for word in reader.readlines():
			stop_words.append(word.strip())
	return stop_words

'''
	函数move_stops： 获得除去停用词的列表
'''
def move_stops(pre_list):
	after_result = []
	stop_words = get_stops()  # 调用函数获得停用词列表
	for word in pre_list:
		if word not in stop_words:
			after_result.append(word)
	return after_result

'''
	函数file_read： 读取文件，调用计算频次函数
		path: 训练文件夹的路径
		sign： 0表示正常邮件， 1表示垃圾邮件
'''
def file_read(path, sign):
	files = os.listdir(path)
	total_mail[sign] = len(files)				#对应总类的邮件数量
	for name in files:
		file = os.path.join(path, name)     	# 单个文件的完整路径
		count_num(file, sign)					# 调用函数计算频次

'''
	函数count_num： 一份邮件中所有单词； 统计一个单词在两类邮件的频次
'''
def count_num(file, sign):
	with open(file, 'r', encoding='gbk') as reader:
		rule = re.compile(r"[^\u4e00-\u9fa5]")		# 过滤掉非中文字符
		line = reader.read()
		content = rule.sub('', line)
		initial_words = jieba.lcut(content)  		# 获得初始词汇列表
		processed_words = move_stops(initial_words) # 删除其中的停用词

	words = list(set(processed_words))        		# 去掉列表中的重复值
	for w in words:
		if w not in word_to_id_map.keys():
			vocabulary.append([0, 0])               # 在词汇表中新增一个位置记录两个频数
			word_to_id_map[w] = len(vocabulary)-1   # 在映射map里记录好单词w和id的对应关系
		vocabulary[ word_to_id_map[w] ][sign] +=1   # 当前单词在sign分类下频数+1

'''
	函数get_frequency：获得每个单词的两个频率
'''
def get_frequency():
	for i in range(0, len(vocabulary)):
		if vocabulary[i][0] == 0.0:     #拉普拉斯平滑，防止出现0
			vocabulary[i][0] = 1		#假设每一个都至少出现一次
		if vocabulary[i][1] == 0.0:
			vocabulary[i][1] = 1
		vocabulary[i][0] = vocabulary[i][0] / total_mail[0]
		vocabulary[i][1] = vocabulary[i][1] / total_mail[1]

'''
	函数save_data：将模型的参数保存到本地, 实现持久化
'''
def save_data():
	file_vocabulary = '../data/word_probability.txt'	# 命名文件名
	with open(file_vocabulary, 'w') as writer:          # 若不存在则新建后打开
		for i in range(0, len(vocabulary)):				# 写入每个单词的两个频率
			writer.write(str(vocabulary[i][0]) + '\t' + str(vocabulary[i][1]) + '\n')

	file_id_map = '../data/word_set.txt'
	with open(file_id_map, 'w') as writer:
		for key, value in word_to_id_map.items():
			writer.write(str(key) + '\t' + str(value) + '\n')

if __name__ == "__main__" :
	file_read('../data/ham', 0)
	file_read('../data/spam', 1)
	get_frequency()
	save_data()
	print('Trained Successfully!')
