import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def validate(model, opt):
    data_loader = create_dataloader(opt)
    print (opt.dataroot)
    dataset_name = opt.dataroot.split('/')[-1]
    print (dataset_name)
    with torch.no_grad():
        y_true, y_pred = [], []

        real_preds = []
        fake_preds = []

        for idx, (img, label) in enumerate(data_loader):

            in_tens = img.cuda()

            model_preds, feats = model(in_tens)
            feats = feats.detach().cpu().numpy()
            preds = model_preds.detach().cpu().numpy()


            if label.item() == 0:
                real_preds.append(feats)
            else:
                fake_preds.append(feats)

            y_pred.extend(model_preds.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())




    encoded_tensors = np.vstack([real_preds, fake_preds])
    encoded_tensors = np.squeeze(encoded_tensors, axis = 1)
    real_labels = np.expand_dims(np.zeros(len(real_preds)), axis = 0)
    fake_labels = np.expand_dims(np.ones(len(fake_preds)), axis = 0)
    labels = np.concatenate([ real_labels, fake_labels  ], axis = 1)
    labels = np.squeeze(labels, axis = 0)

    X_embedded = TSNE().fit_transform(encoded_tensors)
    color_dict = {0: 'r', 1: 'b'}
    for label in np.unique(labels):

        label_idxs = (label == labels)
        color = color_dict[label]
        these_pts = X_embedded[label_idxs]

        xs = these_pts[:,0]
        ys = these_pts[:,1]

        colors = [color] * len(ys)
        if label == 0:
            label_plt = 'Real'
        else:
            label_plt = 'Synthetic'
        plt.scatter(xs, ys, c = colors, label=label_plt)
        plt.legend(loc = 'best')
        plt.xticks([]),plt.yticks([])


    plt.savefig('{}.png'.format(dataset_name))
    plt.clf()



    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
