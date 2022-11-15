# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/15 21:09
 @name: 
 @desc:
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from data_load.dataset import EEGImagesDataset
from model.conv_transformer import ConvTransformer
torch.manual_seed(1234)
np.random.seed(1234)

batch_size = 64
learning_rate = 0.002
epochs = 15

dataset = EEGImagesDataset(path='E:/Datasets/Stanford_digital_repository/img_pkl')
total_x = dataset.__len__()
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=6, shuffle=True)

model = ConvTransformer(num_classes=6, channels=8, num_heads=2, E=16, F=256, T=32, depth=2).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.98), eps=1e-9)

if __name__ == '__main__':
    step = 0
    global_step = 0
    for epoch in range(epochs + 1):
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            model.train()
            optimizer.zero_grad()
            y_ = model(x)
            loss = torch.nn.functional.cross_entropy(y_, y)
            loss.backward()
            optimizer.step()

            step += 1
            global_step += 1
            if step % 50 == 0:
                corrects = (torch.argmax(y_, dim=1).data == y.data)
                accuracy = corrects.cpu().int().sum().numpy() / batch_size
                print('epoch:{}/{} step:{}/{} global_step:{} '
                      'loss={:.5f} acc={:.3f}'.format(epoch, epochs, step, int(total_x / batch_size), global_step, loss,
                                                      accuracy))
        step = 0
    print('done')

