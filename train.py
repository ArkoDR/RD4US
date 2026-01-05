import os
import random
import torch
import argparse
import numpy as np
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from dataset import USDataset
from evaluate import evaluation
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, metavar="", help="Path of dataset.")
parser.add_argument("--save_dir", type=str, required=True, metavar="", help="Path of model.")
parser.add_argument("-e", "--epoch", type=int, default=100, metavar="", help="")
parser.add_argument("-b", "--batch_size", type=int, default=8, metavar="", help="")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, metavar="", help="")
parser.add_argument("-s", "--size", type=int, default=256, metavar="", help="")
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_fucntion_con(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(a[item].shape[0], -1)))
    return loss


def loss_function_adv(a, b, c, d):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        cos1 = cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1))
        cos2 = cos_loss(c[item].view(c[item].shape[0], -1), d[item].view(d[item].shape[0], -1))
        temp = (1 - cos1) + 0.5 * torch.clamp(cos2 - cos1 + 0.2, min=0)
        loss += torch.mean(temp)
    return loss


def train():
    print()
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    size = args.size
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_path = args.input_dir
    ckp_path = args.save_dir
    train_data = USDataset(root=data_path,size=size)
    print(train_data.__len__())
    test_data = USDataset(root=data_path, mode="test",size=size)
    print(test_data.__len__())
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn_layer = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn_layer = bn_layer.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn_layer.parameters()), lr=learning_rate, betas=(0.5,0.999))

    bar = tqdm.tqdm(range(epochs))
    best = 0.5
    for epoch in bar:
        bar.set_description('epoch {}'.format(epoch + 1))
        if args.train_encoder:
            encoder.train()
        else:
            encoder.eval()
        bn_layer.train()
        decoder.train()
        loss_list = []
        for pic_a, pic_e, pic_k, _ in train_dataloader:
            pic_a = pic_a.to(device)
            pic_e = pic_e.to(device)
            pic_k = pic_k.to(device)
            inputs_a = encoder(pic_a, args)
            inputs_e = encoder(pic_e, args)
            inputs_k = encoder(pic_k, args)

            inputs_a_e = inputs_a[:1] + inputs_e[1:]
            inputs_e_a = inputs_e[:1] + inputs_a[1:]
            inputs_a_k = inputs_a[:1] + inputs_k[1:]
            inputs_e_k = inputs_e[:1] + inputs_k[1:]

            bn_a_e = bn_layer(inputs_a_e)
            bn_e_a = bn_layer(inputs_e_a)
            bn_a_k = bn_layer(inputs_a_k)
            bn_e_k = bn_layer(inputs_e_k)

            outputs_a_e = decoder(bn_a_e)
            outputs_e_a = decoder(bn_e_a)
            outputs_a_k = decoder(bn_a_k)
            outputs_e_k = decoder(bn_e_k)

            loss = 0
            loss += loss_fucntion_con(outputs_a_e + outputs_a_k, outputs_e_a + outputs_e_k)
            loss += loss_function_adv(inputs_a + inputs_e, outputs_a_e + outputs_e_a, inputs_a + inputs_e, outputs_a_k + outputs_e_k)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        bar.write('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % 5 == 0:
            res = evaluation(encoder, bn_layer, decoder, test_dataloader, device, False, True)
            metrics = res["metrics"]
            acc, spe, sen, ppv, npv = metrics
            print('Accuracy{:.4f}, Specificity{:.4f}, Sensitivity(Recall){:.4f}, PPV(Precision){:.4f}, NPV{:.4f}, AUC{:.4f}'.format(acc, spe, sen, ppv, npv, res["roc_auc"]))
            if acc > best:
                best = acc
                if not os.path.exists(ckp_path):
                    os.makedirs(ckp_path)
                torch.save({'bn': bn_layer.state_dict(),
                            'decoder': decoder.state_dict()}, ckp_path + f"/Best.pth")
                print("Best model saved.")
    return best


if __name__ == "__main__":
    setup_seed(1125)
    best = train()
    print()
    print(best)