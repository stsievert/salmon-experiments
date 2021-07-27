import json
import numpy
from scipy.linalg import norm
from datetime import datetime
import time

filename = 'fruit_experiment2'
fid = open(filename+'/participants.json')
raw_data = fid.read()
data = eval(raw_data)
participant_list = data['participant_responses']

all_data_by_alg = {}
for participant in participant_list.values():
	for query in participant:
		if 'index_winner' in query:
			index_winner = query['index_winner']
			alg_label = query['alg_label']
			query_timestamp = query['timestamp_query_generated']
			query_dateteime = datetime.strptime(query_timestamp,'%Y-%m-%d %H:%M:%S.%f')

			targets = query['target_indices'] 
			for target in targets:
				if target['label'] == 'center':
					index_center = target['index']
				elif target['label'] == 'left':
					index_left = target['index']
				elif target['label'] == 'right':
					index_right = target['index']

			q = [index_left,index_right,index_center]
			if index_winner==index_right:
				q = [index_right,index_left,index_center]

			if alg_label not in all_data_by_alg:
				all_data_by_alg[alg_label] = []

			all_data_by_alg[alg_label].append( (query_dateteime,q) )


for alg_label in all_data_by_alg:
	all_data_by_alg[alg_label] = sorted(all_data_by_alg[alg_label], key=lambda item: item[0])

n = 30 
d = 2
batch_size = 300
passes_over_data = 64
num_neighbors = 3
import utilsMDS
import utilsCrowdKernel
import utilsSTE
embed_modules = [('Hinge',utilsMDS),('CK',utilsCrowdKernel),('tSTE',utilsSTE)]

# find nearest neighbors of each point
target_mapping = [(0, 'strangefruit30/i0126.png'), (1, 'strangefruit30/i0208.png'), (2, 'strangefruit30/i0076.png'), (3, 'strangefruit30/i0326.png'), (4, 'strangefruit30/i0526.png'), (5, 'strangefruit30/i0322.png'), (6, 'strangefruit30/i0312.png'), (7, 'strangefruit30/i0036.png'), (8, 'strangefruit30/i0414.png'), (9, 'strangefruit30/i0256.png'), (10, 'strangefruit30/i0074.png'), (11, 'strangefruit30/i0050.png'), (12, 'strangefruit30/i0470.png'), (13, 'strangefruit30/i0022.png'), (14, 'strangefruit30/i0430.png'), (15, 'strangefruit30/i0254.png'), (16, 'strangefruit30/i0572.png'), (17, 'strangefruit30/i0200.png'), (18, 'strangefruit30/i0524.png'), (19, 'strangefruit30/i0220.png'), (20, 'strangefruit30/i0438.png'), (21, 'strangefruit30/i0454.png'), (22, 'strangefruit30/i0112.png'), (23, 'strangefruit30/i0494.png'), (24, 'strangefruit30/i0194.png'), (25, 'strangefruit30/i0152.png'), (26, 'strangefruit30/i0420.png'), (27, 'strangefruit30/i0142.png'), (28, 'strangefruit30/i0114.png'), (29, 'strangefruit30/i0184.png')]
real_target_mapping = [ (x[0],int(x[1].split('/i')[1].split('.')[0])) for x in target_mapping]
# target mapping is of format (index,position_on_line)
nearest_neighbor_lookup = numpy.zeros(n).tolist()
for this_item in real_target_mapping:
	tmp = sorted(real_target_mapping, key=lambda item: numpy.abs(item[1]-this_item[1]))
	print str(this_item) + ':' + str(tmp[1]) + ',' + str(tmp[2])
	nearest_neighbor_lookup[this_item[0]] = tmp[1][0]


S_test = [ item[1] for item in all_data_by_alg['Test'] ]

nn_acc = {}
losses = {}
train_losses = {}
embeddings = {}
num_answers = {}
algorithms = []
verbose = False
for alg_label in all_data_by_alg:
	if alg_label != 'Test':
		algorithms.append(alg_label)

		for embed_strategy in embed_modules:
			embed_name = embed_strategy[0]
			embed_module = embed_strategy[1]

			nn_acc[(embed_name,alg_label)] = []
			losses[(embed_name,alg_label)] = []
			train_losses[(embed_name,alg_label)] = []
			embeddings[(embed_name,alg_label)] = []
			num_answers[(embed_name,alg_label)] = []

			S = [ item[1] for item in all_data_by_alg[alg_label] ]

			for k in range(1,len(S)/batch_size+2):
				ts = time.time()

				if k*batch_size > len(S):
					sample_size = len(S)
				else:
					sample_size = k*batch_size

				X,loss = embed_module.computeEmbeddingWithEpochSGD(n,d,S[:sample_size],max_num_passes=passes_over_data,epsilon=0.,verbose=verbose)
				X,a,b,c = embed_module.computeEmbeddingWithGD(X,S[:sample_size],max_iters=passes_over_data,epsilon=0.,verbose=verbose)
				train_emp_loss,hinge_loss = utilsMDS.getLoss(X,S[:sample_size])

				# compute triplet preduction loss on test set
				emp_loss,hinge_loss_old = utilsMDS.getLoss(X,S_test)

				# compute nearest neighbor accuracy
				acc = 0.
				for this_i in range(n):
					tmp = sorted(range(n), key=lambda that_i: norm(X[that_i]-X[this_i])  )
					this_nn = nearest_neighbor_lookup[this_i]

					for ell in range(1,num_neighbors+1):
						acc += 1.0*(tmp[ell]==this_nn)
				acc = acc / n

				nn_acc[(embed_name,alg_label)].append(acc)
				losses[(embed_name,alg_label)].append(emp_loss)
				train_losses[(embed_name,alg_label)].append(train_emp_loss)
				embeddings[(embed_name,alg_label)].append(X)
				num_answers[(embed_name,alg_label)].append(sample_size)

				print str( (embed_name,alg_label) ) + " : " + str(sample_size) + "   " + str(time.time()-ts)


print nn_acc
print num_answers
print losses
embed_names = [ embed_strategy[0] for embed_strategy in embed_modules ]

# import matplotlib.pyplot as plt

# colors = ['b','g','r','c']
# linetypes = ['-','--',':']
# legend_labels = []

# min_x = float('inf')
# max_x = -float('inf')
# for idx1,alg_label in enumerate(algorithms):
# 	for idx2,embed_name in enumerate(embed_names):
# 		if min(num_answers[(embed_name,alg_label)])<min_x:
# 			min_x = min(num_answers[(embed_name,alg_label)])
# 		if max(num_answers[(embed_name,alg_label)])>max_x:
# 			max_x = max(num_answers[(embed_name,alg_label)])


# plt.figure()
# for idx1,alg_label in enumerate(algorithms):
# 	for idx2,embed_name in enumerate(embed_names):
# 		plt.plot(num_answers[(embed_name,alg_label)],losses[(embed_name,alg_label)],colors[idx1]+linetypes[idx2])
# 		legend_labels.append( alg_label+':'+embed_name )

# plt.xlim((min_x,max_x))
# plt.ylim((.14,.26))
# plt.grid(True)
# plt.xlabel('Observed Triplets')
# plt.ylabel('Triplet prediction error')

# plt.figure()
# for idx1,alg_label in enumerate(algorithms):
# 	for idx2,embed_name in enumerate(embed_names):
# 		plt.plot(num_answers[(embed_name,alg_label)],nn_acc[(embed_name,alg_label)],colors[idx1]+linetypes[idx2])
# 		legend_labels.append( alg_label+':'+embed_name )

# plt.xlim((min_x,max_x))
# plt.grid(True)
# plt.xlabel('Observed Triplets')
# plt.ylabel('Nearest Neighbor Accuracy')
# # plt.legend(legend_labels)
# # plt.legend(legend_labels, loc=3,ncol=4, mode="expand", borderaxespad=0.)

# plt.show()

import pickle
pickled_data = {}
pickled_data['algorithms'] = algorithms
pickled_data['embed_names'] = embed_names
pickled_data['nn_acc'] = nn_acc
pickled_data['num_answers'] = num_answers
pickled_data['losses'] = losses
pickled_data['train_losses'] = train_losses
pickled_data['embeddings'] = embeddings
full_filename = filename+'_'+str(datetime.now())+'_'+'_pickled_data.pkl'
pickle.dump(pickled_data, open(full_filename, "wb"))
print '########    ' + full_filename + '    #########'




from boto.s3.connection import S3Connection
from boto.s3.key import Key
import boto
import os
AWS_ACCESS_ID = os.environ.get('AWS_ACCESS_ID', '')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
conn = S3Connection(AWS_ACCESS_ID,AWS_SECRET_ACCESS_KEY)
b = conn.get_bucket('fruit-experiment2-results-120eu01h3108eh2108eh')

print AWS_ACCESS_ID
print AWS_SECRET_ACCESS_KEY

while True:
	try:
		print "trying to save " + str(full_filename)
		k = Key(b)
		k.key = str(full_filename)
		bytes_saved = k.set_contents_from_filename( str(full_filename) )
		break
		# bytes_saved = k.set_contents_from_string(pickle_string)
	except:
		print "FAILED!"
		pass

print "[ %s ] done with backup of file %s to S3...  %d bytes saved" % (str(datetime.now()),full_filename,bytes_saved)









# pickled_data = pickle.load(open('fruit_experiment1_2015-06-02 09:35:07.368078__pickled_data.pkl','rb'))
algorithms = pickled_data['algorithms']
embed_names = pickled_data['embed_names']
nn_acc = pickled_data['nn_acc']
num_answers = pickled_data['num_answers']
losses = pickled_data['losses'] 


# for vec in losses:
# 	plt.plot(num_answers[vec],losses[vec])
# plt.xlabel('Observed Triplets')
# plt.ylabel('Test error')
# plt.legend(losses.keys())
# plt.show()
