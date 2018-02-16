import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from sine import sine_generator
from wavenet import WaveNet

g = sine_generator(seq_size=2200, mu=64)
net = WaveNet(n_out=64, n_residue=24, n_skip=128, dilation_depth=10, n_layers=2)
optimizer = optim.Adam(net.parameters(), lr=0.01)
batch_size = 64

loss_save = []
max_epoch = 2000
for epoch in range(max_epoch):
    optimizer.zero_grad()
    loss = 0
    # iterate over the data
    for _ in range(batch_size):
        batch = next(g)
        x = batch[:-1]
        logits = net(x)
        sz = logits.size(0)
        loss = loss + nn.functional.cross_entropy(logits, batch[-sz:])
    loss = loss / batch_size
    loss.backward()
    optimizer.step()
    loss_save.append(loss.data[0])
    # monitor progress
    if epoch % 100 == 0:
        batch = next(g)
        print('epoch %i, loss %.4f' % (epoch, loss.data[0]))
        logits = net(batch[:-1])
        _, i = logits.max(dim=1)
        plt.figure(figsize=[16, 4])
        plt.plot(i.data.tolist())
        plt.plot(batch.data.tolist()[sum(net.dilations) + 1:])
        plt.legend(['generated', 'data'])
        plt.title('epoch %i' % epoch)
        plt.tight_layout()
        plt.savefig('train/epoch_%i.png' % epoch)

y_gen = net.generate(batch, 4000)
plt.figure(figsize=[16, 4])
plt.plot(y_gen, '--', c='b')
plt.plot(batch.data.tolist(), ms=2, c='k')
plt.legend(['generated', 'data'])
plt.savefig('train/generation.png')

torch.save(net.state_dict(), 'train/wavenet.pt')
