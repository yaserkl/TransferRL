import sys, os
from glob import glob

datasets = glob("{}/*".format(sys.argv[1]))
for ds in datasets: # generalized RLSeq2Seq
	exps = glob("{}/*".format(ds))
	for exp in exps: # cnn_generalized_pretrain  cnn_main  main
		ind = open("{}/train/checkpoint".format(exp)).readlines()[1].strip().split('-')[-1]
		try:
			ind = int(ind[0:-1]) # remove "
		except:
			ind = int(ind[0:-1].split('_')[0]) # remove "
		delete_list = glob("{}/train/model.ckpt-*".format(exp))
		for del_item in delete_list:
			current_step = del_item.split(".")[1].split("-")[1]
			try:
				current_step = int(del_item.split(".")[1].split("-")[1])
			except:
				current_step = int(del_item.split(".")[1].split("-")[1].split('_')[0])
			if current_step < ind:
				os.remove(del_item)
