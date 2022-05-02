
'''混合精度训练对大模型适用，对于小模型则很差，以下是使用模板
注意：amp有两种方式：用pytorch内置的torch.cuda.amp，但是需要pytorch1.7及以上，但是安装的是1.5.1版本的，
所以就使用第二种：apex.amp'''


import torch
from apex import amp
import time


N, D_in, D_out = 64, 1024, 512
x = torch.randn(N, D_in, device='cuda')
y = torch.randn(N, D_out, device='cuda')
model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


fp16 = True
save_test = True

if fp16: #这一步只能定义在训练的外面，不能定义在训练的循环内部。因此如果想用apex的话，不能直接在fit_one_epoch函数内直接修改，应该在train.py的训练循环外面先修改下面这一步
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

t1 = time.time()

print('start')
for t in range(5000):

    optimizer.zero_grad()
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    if fp16:
        #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")，放在这儿会报错
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()

    if t == 4999 and save_test:
        torch.save(model, 'model_epoch_4999.pth')

t2 = time.time()
print('finish,time cost {}'.format(t2-t1))


'''github介绍的使用方式见：https://github.com/NVIDIA/apex

# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# Train your model
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
...

# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'amp': amp.state_dict()
}
torch.save(checkpoint, 'amp_checkpoint.pt')
...

# Restore  --中断之后恢复训练
model = ...
optimizer = ...
checkpoint = torch.load('amp_checkpoint.pt')

model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
amp.load_state_dict(checkpoint['amp'])

# Continue training
......

'''
