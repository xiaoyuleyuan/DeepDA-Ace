
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import torch
from torch.autograd import Variable
from models import main_models
from torchsummary import summary
import numpy as np
import datasets
from datasets import dataset
import sklearn
from sklearn import metrics
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

parser=argparse.ArgumentParser()
parser.add_argument('--n_epoches_1',type=int,default= 50)
parser.add_argument('--n_epoches_2',type=int,default= 100)
parser.add_argument('--n_epoches_3',type=int,default=200)
parser.add_argument('--n_target_samples',type=int,default=7)
parser.add_argument('--batch_size',type=int,default= 256)

opt=vars(parser.parse_args())

use_cuda=True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

    
#--------------pretrain g and h for step 1---------------------------------
species_all = ['Homo_sapiens','Rattus_norvegicus','Schistosoma_japonicum','Saccharomyces_cerevisiae','Mus_musculus','Escherichia_coli','Bacillus_velezensis','Plasmodium_falciparum','Oryza_sativa','Arabidopsis_thaliana']
specie1 = species_all[0]
specie2 = species_all[1]

result_path = '/harddisk/hdd_d/liuyu/PTM/ace/PTM_GAN/results/'
if not os.path.exists(result_path):
        os.makedirs(result_path)
        
model_path = '/harddisk/hdd_d/liuyu/PTM/ace/PTM_GAN/results/'
if not os.path.exists(model_path):
        os.makedirs(model_path)
        
mainfolder = '/harddisk/hdd_d/liuyu/PTM/ace/PTM_GAN/result/GAN/' + specie2 + '/' 
if not os.path.exists(mainfolder):
    os.makedirs(mainfolder)
figfolder = mainfolder + '/figure'
if not os.path.exists(figfolder):
    os.makedirs(figfolder)
prefolder = mainfolder + '/prediction'
if not os.path.exists(prefolder):
    os.makedirs(prefolder)
feafolder = mainfolder + '/feature'
if not os.path.exists(feafolder):
    os.makedirs(feafolder)
modelfolder = mainfolder + '/model'
if not os.path.exists(modelfolder):
    os.makedirs(modelfolder)

data_path = '/harddisk/hdd_d/liuyu/PTM/ace/PTM_GAN/data_npy/'
batch_size=opt['batch_size']
test_CRC_data = dataset(data_path,specie1, mode = 'test')
test_data_loader = torch.utils.data.DataLoader(test_CRC_data, batch_size=batch_size, shuffle=True) 

train_CRC_data = dataset(data_path,specie1, mode = 'train')
train_data_loader = torch.utils.data.DataLoader(train_CRC_data, batch_size=batch_size, shuffle=True) 
validation_CRC_data = dataset(data_path,specie1, mode = 'valid')
validation_data_loader = torch.utils.data.DataLoader(validation_CRC_data, batch_size=batch_size, shuffle=True) 

def plot_loss(y):

    x = range(0,len(y))
    plt.plot(x, y, '.-',color="red")
    plt_title = 'xxx'
    plt.title(plt_title)
    plt.xlabel('per 200 times')
    plt.ylabel('LOSS')
    plt.savefig(result_path+'ptm-{:s}-loss.png'.format('HUMAN'))
    
def save_feature(feature,phase,index):
    l = feature.shape[1]
    num = feature.shape[0]
    final_matrix = np.ones((num,l))
    for i in range(num):
        temp = feature[i].flatten()
        final_matrix[i] = temp
    np.savetxt(feafolder + '/feature_{:s}_{:d}.txt'.format(phase,index),final_matrix)
    
    
classifier=main_models.Classifier()
encoder=main_models.Encoder()
discriminator=main_models.DCD(input_features=128)

#print('net--------------------------')
#print(encoder)

classifier.to(device)
encoder.to(device)
discriminator.to(device)
loss_fn=torch.nn.CrossEntropyLoss()
summary(encoder,(1, 31, 21))
exit(0)
optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.0001,weight_decay=0.0001)

Loss = []
for epoch in range(opt['n_epoches_1']):
    if epoch == 50:
        optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.00002,weight_decay=0.0001)
    for i, (data, labels) in enumerate(train_data_loader):
        data=data.to(device)
        labels = labels[:,1]
        labels=labels.to(device)
        data = Variable(data,requires_grad=True)
        labels = Variable(labels)
        optimizer.zero_grad()
        y_pred=classifier(encoder(data))
        labels = labels.long()
        loss=loss_fn(y_pred,labels)
        loss.backward()
        optimizer.step()
        
    acc=0
    auc = 0
    y_test_pred_all = 0
    labels_all = 0
    for i, (data, labels) in enumerate(test_data_loader):
        
        data=data.to(device)
        labels = labels[:,1]
        labels=labels.to(device)
        labels = labels.long()
        y_test_pred=classifier(encoder(data))
        loss=loss_fn(y_test_pred,labels)
        running_loss = loss.item()
        Loss.append(running_loss)
        
        if i == 0:
            y_test_pred_all = y_test_pred.detach().cpu()
            labels_all = labels.detach().cpu()
        else:
            y_test_pred_all = torch.cat((y_test_pred_all,y_test_pred.detach().cpu()), 0)
            labels_all = torch.cat((labels_all,labels.detach().cpu()), 0)        

        acc+=(torch.max(y_test_pred.detach().cpu(),1)[1]==labels.detach().cpu()).float().mean().item()
        
    #y_test_pred_all = y_test_pred_all.detach().cpu()
    #labels_all = labels_all.detach().cpu()
    #print(labels_all)
    plot_loss(Loss)
    accuracy=round(acc / float(i+1), 3)
    auc = metrics.roc_auc_score(labels_all,y_test_pred_all[:, 1])
    print("step1----Epoch %d/%d  test accuracy: %.3f "%(epoch+1,opt['n_epoches_1'],accuracy))
    print("step1----Epoch %d/%d  test auc : %.3f "%(epoch+1,opt['n_epoches_1'],auc))
    
    fpr_t, tpr_t, _ = metrics.roc_curve(labels_all,y_test_pred_all[:, 1])
    fig, ax = plt.subplots()
    ax.plot(fpr_t, tpr_t, 'b-', label='CNN-test {:.3%}'.format(auc))
    ax.legend(loc='lower right', shadow=True)
    plt.title('test of {:s}'.format('HUMAN'))
    #plt.savefig(result_path+'ptm-{:s}-epoch-{:d}-auc-{:.4f}-acc-{:.4f}-test.png'.format('HUMAN', epoch+1, auc, accuracy))
    plt.close()
    torch.save(encoder, model_path + 'encoder1.pth.tar') 
    torch.save(classifier, model_path + 'classifier1.pth.tar') 
               
    """
    acc = 0
    auc = 0  
    y_test_pred_all = 0
    labels_all = 0
    for i, (data, labels) in enumerate(train_data_loader):
        
        data=data.to(device)
        labels=labels.to(device)
        y_test_pred=classifier(encoder(data))
        if i == 0:
            y_test_pred_all = y_test_pred
            labels_all = labels
        else:
            y_test_pred_all = torch.cat((y_test_pred_all,y_test_pred), 0)
            labels_all = torch.cat((labels_all,labels), 0)        
        acc+=(torch.max(y_test_pred,1)[1]==labels[:,1]).float().mean().item()
    y_test_pred_all = y_test_pred_all.detach().numpy()
    labels_all = labels_all.detach().numpy()

    accuracy=round(acc / float(i), 3)
    auc = metrics.roc_auc_score(labels_all[:, 1],y_test_pred_all[:, 1])
    print("step1----Epoch %d/%d  train accuracy: %.3f "%(epoch+1,opt['n_epoches_1'],accuracy))
    print("step1----Epoch %d/%d  train auc: %.3f "%(epoch+1,opt['n_epoches_1'],auc))
    """
    
X_s,Y_s=datasets.sample_data(data_path,specie1, mode = 'train')
X_t,Y_t=datasets.create_target_samples(data_path,specie2, mode = 'train')  
#-----------------train DCD for step 2--------------------------------
optimizer_D=torch.optim.Adam(discriminator.parameters(),lr=0.01)

for epoch in range(opt['n_epoches_2']):
    # data
    groups,aa = datasets.sample_groups(X_s,Y_s,X_t,Y_t,seed=epoch)

    n_iters = 4 * len(groups[1])
    
    
    index_list = torch.randperm(n_iters)
    mini_batch_size=40 #use mini_batch train can be more stable

    loss_mean=[]

    X1=[];X2=[];ground_truths=[]
    for index in range(n_iters):

        ground_truth=index_list[index]//len(groups[1])
        x1,x2=groups[ground_truth][index_list[index]-len(groups[1])*ground_truth]
        X1.append(x1)
        X2.append(x2)
        ground_truths.append(ground_truth)
        
        #select data for a mini-batch to train
        if (index+1)%mini_batch_size==0:
            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths=torch.LongTensor(ground_truths)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths=ground_truths.to(device)
            optimizer_D.zero_grad()
            X_cat=torch.cat([encoder(X1),encoder(X2)],1)
            y_pred=discriminator(X_cat.detach())
            
            loss=loss_fn(y_pred,ground_truths)
            loss.backward()
            optimizer_D.step()
            loss_mean.append(loss.item())
            X1 = []
            X2 = []
            ground_truths = []

    print("step2----Epoch %d/%d loss:%.3f"%(epoch+1,opt['n_epoches_2'],np.mean(loss_mean)))



encoder=torch.load(model_path + 'encoder1.pth.tar')
classifier=torch.load(model_path + 'classifier1.pth.tar')
#-------------------training for step 3-------------------
optimizer_g_h=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.00001)
optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=0.00000001)


test_CRC_data = dataset(data_path,specie2, mode = 'test')
test_data_loader = torch.utils.data.DataLoader(test_CRC_data, batch_size=batch_size, shuffle=True) 

for epoch in range(opt['n_epoches_3']):
    #---training g and h , DCD is frozen

    groups, groups_y = datasets.sample_groups(X_s,Y_s,X_t,Y_t,seed=opt['n_epoches_2']+epoch)
    G1, G2, G3, G4 = groups
    Y1, Y2, Y3, Y4 = groups_y
    groups_2 = [G2, G4]
    groups_y_2 = [Y2, Y4]

    n_iters = 2 * len(G2)
    index_list = torch.randperm(n_iters)

    n_iters_dcd = 4 * len(G2)
    index_list_dcd = torch.randperm(n_iters_dcd)

    mini_batch_size_g_h = 20 #data only contains G2 and G4 ,so decrease mini_batch
    mini_batch_size_dcd= 40 #data contains G1,G2,G3,G4 so use 40 as mini_batch
    X1 = []
    X2 = []
    ground_truths_y1 = []
    ground_truths_y2 = []
    dcd_labels=[]
    for index in range(n_iters):


        ground_truth=index_list[index]//len(G2)
        x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        # y1=torch.LongTensor([y1.item()])
        # y2=torch.LongTensor([y2.item()])
        
        dcd_label=0 if ground_truth==0 else 2
        X1.append(x1)
        X2.append(x2)
        ground_truths_y1.append(y1)
        ground_truths_y2.append(y2)
        dcd_labels.append(dcd_label)
        
       
        if (index+1)%mini_batch_size_g_h==0:
            ground_truths_y1 = torch.stack(ground_truths_y1)
            ground_truths_y2 = torch.stack(ground_truths_y2)
            ground_truths_y1 = ground_truths_y1[:,1]
            ground_truths_y2 = ground_truths_y2[:,1]
            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths_y1=torch.Tensor(ground_truths_y1)
            ground_truths_y2 = torch.Tensor(ground_truths_y2)
            dcd_labels=torch.LongTensor(dcd_labels)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths_y1=ground_truths_y1.to(device)
            ground_truths_y2 = ground_truths_y2.to(device)
            dcd_labels=dcd_labels.to(device)

            optimizer_g_h.zero_grad()

            encoder_X1=encoder(X1)
            encoder_X2=encoder(X2)

            X_cat=torch.cat([encoder_X1,encoder_X2],1)
            y_pred_X1=classifier(encoder_X1)
            y_pred_X2=classifier(encoder_X2)
            y_pred_dcd=discriminator(X_cat)
            ground_truths_y1 = ground_truths_y1.long()
            ground_truths_y2 = ground_truths_y2.long()
            loss_X1=loss_fn(y_pred_X1,ground_truths_y1)
            loss_X2=loss_fn(y_pred_X2,ground_truths_y2)
            loss_dcd=loss_fn(y_pred_dcd,dcd_labels)

            loss_sum = loss_X1 + loss_X2 + 0.2 * loss_dcd

            loss_sum.backward()
            optimizer_g_h.step()

            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []


    #----training dcd ,g and h frozen
    X1 = []
    X2 = []
    ground_truths = []
    for index in range(n_iters_dcd):

        ground_truth=index_list_dcd[index]//len(groups[1])

        x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
        X1.append(x1)
        X2.append(x2)
        ground_truths.append(ground_truth)

        if (index + 1) % mini_batch_size_dcd == 0:
            X1 = torch.stack(X1)
            X2 = torch.stack(X2)
            ground_truths = torch.LongTensor(ground_truths)
            X1 = X1.to(device)
            X2 = X2.to(device)
            ground_truths = ground_truths.to(device)

            optimizer_d.zero_grad()
            X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
            y_pred = discriminator(X_cat.detach())
            loss = loss_fn(y_pred, ground_truths)
            loss.backward()
            optimizer_d.step()
            # loss_mean.append(loss.item())
            X1 = []
            X2 = []
            ground_truths = []

    #testing
    acc=0
    auc = 0
    y_test_pred_all = 0
    labels_all = 0
    for i, (data, labels) in enumerate(test_data_loader):
        
        data=data.to(device)
        labels = labels[:,1]
        labels=labels.to(device)
        labels= labels.long()
        y_test_pred=classifier(encoder(data))
        loss=loss_fn(y_test_pred,labels)
        running_loss = loss.item()
        Loss.append(running_loss)
        plot_loss(Loss)
        if i == 0:
            y_test_pred_all = y_test_pred.detach().cpu()
            labels_all = labels.detach().cpu()
        else:
            y_test_pred_all = torch.cat((y_test_pred_all,y_test_pred.detach().cpu()), 0)
            labels_all = torch.cat((labels_all,labels.detach().cpu()), 0)        
        a =torch.max(y_test_pred.detach().cpu(),1)[1]
        b =torch.max(y_test_pred.detach().cpu(),1)[1]==labels.detach().cpu()
        acc+=(torch.max(y_test_pred.detach().cpu(),1)[1]==labels.detach().cpu()).float().mean().item()
    accuracy = round(acc / float(i+1), 3)  
    auc = metrics.roc_auc_score(labels_all,y_test_pred_all[:, 1])
    np.savetxt(prefolder+ '/prediction scores of epoch-{:d}.txt'.format(epoch),y_test_pred_all)
    np.savetxt(prefolder+ '/label of epoch-{:d}.txt'.format(epoch),labels_all)
    
    print("step1----Epoch %d/%d  accuracy: %.3f "%(epoch+1,opt['n_epoches_3'],accuracy))
    print("step1----Epoch %d/%d  auc: %.3f "%(epoch+1,opt['n_epoches_3'],auc))
    torch.save(encoder, modelfolder + '/encoder'+ str(epoch)+'.pth.tar') 
    torch.save(classifier, modelfolder + '/classifier'+ str(epoch)+'.pth.tar') 
    
    fpr_t, tpr_t, _ = metrics.roc_curve(labels_all,y_test_pred_all[:, 1])
    fig, ax = plt.subplots()
    ax.plot(fpr_t, tpr_t, 'b-', label='CNN-test {:.3%}'.format(auc))
    ax.legend(loc='lower right', shadow=True)
    plt.title('test of {:s}'.format(specie2))
    plt.savefig(figfolder + '/ptm-{:s}-epoch-{:d}-auc-{:.4f}-acc-{:.4f}-test.png'.format(specie2, epoch, auc, accuracy))
    
    
    
    
    
    
    
    
    
    
    
    
    
