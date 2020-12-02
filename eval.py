import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *


# Running tests
opt = TestOptions().parse(print_options=False)

model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))

for v_id, val in enumerate(vals):
    if not val =='cyclegan':
        continue

    opt.dataroot = '{}/{}'.format(dataroot, val)
    # opt.dataroot = 'dataset/chest_xray/gan_exp'
    # opt.dataroot = 'dataset/mnist/gan_exp'

    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    # opt.classes = ['']

    opt.no_resize = True    # testing without resizing by default
    print (opt.dataroot)
    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, _, _, _, _ = validate(model, opt)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))


csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
