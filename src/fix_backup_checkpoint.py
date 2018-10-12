import sys
from glob import glob

datasets = glob("{}/*".format(sys.argv[1]))
for ds in datasets:
    exps = glob("{}/*".format(ds))
    for exp in exps:
        models = glob("{}/*".format(exp))
        for model in models:
            try:
                index_files = set([int(_.split('-')[1].split('.')[0]) for _ in glob("{}/train/model.ckpt-*.index".format(model))])
                data_files = set([int(_.split('-')[1].split('.')[0]) for _ in glob("{}/train/model.ckpt-*.data*".format(model))])
                meta_files = set([int(_.split('-')[1].split('.')[0]) for _ in glob("{}/train/model.ckpt-*.meta".format(model))])
                checkpoint = max(index_files.intersection(data_files).intersection(meta_files))
                checkpoint_file = "{}/train/checkpoint".format(model)
                chkp = open(checkpoint_file).readlines()

                current_chkp_num = chkp[0].split("-")[-1].strip()[:-1]
                #chkp_num = chkp[1].split("-")[-1].strip('"').strip()

                with open(checkpoint_file, 'w') as fw:
                    for c in chkp:
                        fw.write(c.replace(current_chkp_num, str(checkpoint)).replace('ubuntu','yaserkl'))
            except:
                continue