import json
import numpy
from scipy.linalg import norm
from datetime import datetime

from boto.s3.connection import S3Connection
from boto.s3.key import Key
import boto
import os
AWS_ACCESS_ID = os.environ.get('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
conn = S3Connection(AWS_ACCESS_ID,AWS_SECRET_ACCESS_KEY)
# b = conn.get_bucket('sim-results-120eu01h3108eh2108eh')
# b = conn.get_bucket('sim-results-120eu01h3108eh2108eh')
b = conn.get_bucket('fruit-experiment1-results-120eu01h3108eh2108eh')

print AWS_ACCESS_ID
print AWS_SECRET_ACCESS_KEY

best_train_losses = {}
best_losses = {}
best_nn_acc = {}
best_embedding = {}
algorithms = {}
embed_names = {}
nn_acc = {}
num_answers = {}
losses = {}

cnt = 0
import pickle
for key in b.list():

	# if cnt > 0:
	# 	break
	# else:
	# 	cnt += 1

	print key.key
	key.get_contents_to_filename('tmp.data')
	pickled_data = pickle.load(open('tmp.data', "rb"))

	algorithms = pickled_data['algorithms']
	embed_names = pickled_data['embed_names']
	nn_acc = pickled_data['nn_acc']
	num_answers = pickled_data['num_answers']
	losses = pickled_data['losses'] 
	train_losses = pickled_data['train_losses'] 
	embeddings = pickled_data['embeddings']

	for idx1,alg_label in enumerate(algorithms):
		for idx2,embed_name in enumerate(embed_names):
			# print losses[(embed_name,alg_label)]
			# print nn_acc[(embed_name,alg_label)]
			for idx3,sample_size in enumerate(losses[(embed_name,alg_label)]):
				try:
					best_train_losses[(embed_name,alg_label)][idx3]
					best_losses[(embed_name,alg_label)][idx3]
					best_nn_acc[(embed_name,alg_label)][idx3]

					# if train_losses[(embed_name,alg_label)][idx3] < best_train_losses[(embed_name,alg_label)][idx3]:
					# 	best_train_losses[(embed_name,alg_label)][idx3] = train_losses[(embed_name,alg_label)][idx3]
					# 	best_losses[(embed_name,alg_label)][idx3] = losses[(embed_name,alg_label)][idx3]
					# 	best_nn_acc[(embed_name,alg_label)][idx3] = nn_acc[(embed_name,alg_label)][idx3]

					if losses[(embed_name,alg_label)][idx3] < best_losses[(embed_name,alg_label)][idx3]:
						best_losses[(embed_name,alg_label)][idx3] = losses[(embed_name,alg_label)][idx3]

					if nn_acc[(embed_name,alg_label)][idx3] > best_nn_acc[(embed_name,alg_label)][idx3]:
						best_nn_acc[(embed_name,alg_label)][idx3] = nn_acc[(embed_name,alg_label)][idx3]

				except:
					best_train_losses[(embed_name,alg_label)] = train_losses[(embed_name,alg_label)]
					best_losses[(embed_name,alg_label)] = losses[(embed_name,alg_label)]
					best_nn_acc[(embed_name,alg_label)] = nn_acc[(embed_name,alg_label)]
			

			if losses[(embed_name,alg_label)][-1] <= best_losses[(embed_name,alg_label)][-1]:
				best_embedding[(embed_name,alg_label)] = embeddings[(embed_name,alg_label)][-1]


losses = best_losses
nn_acc = best_nn_acc


n = 30
target_mapping = [(0, 'strangefruit30/i0126.png'), (1, 'strangefruit30/i0208.png'), (2, 'strangefruit30/i0076.png'), (3, 'strangefruit30/i0326.png'), (4, 'strangefruit30/i0526.png'), (5, 'strangefruit30/i0322.png'), (6, 'strangefruit30/i0312.png'), (7, 'strangefruit30/i0036.png'), (8, 'strangefruit30/i0414.png'), (9, 'strangefruit30/i0256.png'), (10, 'strangefruit30/i0074.png'), (11, 'strangefruit30/i0050.png'), (12, 'strangefruit30/i0470.png'), (13, 'strangefruit30/i0022.png'), (14, 'strangefruit30/i0430.png'), (15, 'strangefruit30/i0254.png'), (16, 'strangefruit30/i0572.png'), (17, 'strangefruit30/i0200.png'), (18, 'strangefruit30/i0524.png'), (19, 'strangefruit30/i0220.png'), (20, 'strangefruit30/i0438.png'), (21, 'strangefruit30/i0454.png'), (22, 'strangefruit30/i0112.png'), (23, 'strangefruit30/i0494.png'), (24, 'strangefruit30/i0194.png'), (25, 'strangefruit30/i0152.png'), (26, 'strangefruit30/i0420.png'), (27, 'strangefruit30/i0142.png'), (28, 'strangefruit30/i0114.png'), (29, 'strangefruit30/i0184.png')]
url_target_mapping = range(n)
for item in target_mapping:
	url_target_mapping[ item[0] ] = "http://next.discovery.s3.amazonaws.com/2015-05-28_" + item[1]
print url_target_mapping

docs = {}
for idx1,alg_label in enumerate(algorithms):
	for idx2,embed_name in enumerate(embed_names):
		docs[(embed_name,alg_label)] = []
		for i in range(n):
			x = best_embedding[(embed_name,alg_label)][i][0]
			y = best_embedding[(embed_name,alg_label)][i][1]
			doc = {'index':i, 'target':{ 'primary_description':url_target_mapping[i], 'primary_type': "image" }, 'x':x,'y':y }
			docs[(embed_name,alg_label)].append(doc)

		print str((embed_name,alg_label))
		print docs[(embed_name,alg_label)]
		print
		print




print num_answers.keys()
print losses.keys()
print nn_acc.keys()

import matplotlib.pyplot as plt

colors = ['b','g','r','c']
linetypes = ['-','--',':']
legend_labels = []

min_x = float('inf')
max_x = -float('inf')
for idx1,alg_label in enumerate(algorithms):
	for idx2,embed_name in enumerate(embed_names):
		if min(num_answers[(embed_name,alg_label)])<min_x:
			min_x = min(num_answers[(embed_name,alg_label)])
		if max(num_answers[(embed_name,alg_label)])>max_x:
			max_x = max(num_answers[(embed_name,alg_label)])


plt.figure()
for idx1,alg_label in enumerate(algorithms):
	for idx2,embed_name in enumerate(embed_names):
		plt.plot(num_answers[(embed_name,alg_label)],losses[(embed_name,alg_label)],colors[idx1]+linetypes[idx2])
		legend_labels.append( alg_label+':'+embed_name )

plt.xlim((min_x,max_x))
plt.ylim((.14,.26))
plt.grid(True)
plt.xlabel('Observed Triplets')
plt.ylabel('Triplet prediction error')

plt.figure()
for idx1,alg_label in enumerate(algorithms):
	for idx2,embed_name in enumerate(embed_names):
		plt.plot(num_answers[(embed_name,alg_label)],nn_acc[(embed_name,alg_label)],colors[idx1]+linetypes[idx2])
		legend_labels.append( alg_label+':'+embed_name )

plt.xlim((min_x,max_x))
plt.grid(True)
plt.xlabel('Observed Triplets')
plt.ylabel('Nearest Neighbor Accuracy')


# plt.legend(legend_labels)
# plt.legend(legend_labels, loc=3,ncol=4, mode="expand", borderaxespad=0.)

plt.show()

