import warnings
warnings.simplefilter('ignore')
from sklearn.metrics import confusion_matrix
from net.model import *
import util
from torch.utils.data import DataLoader
import torch
import numpy as np
from random import randrange
argv = util.option.parse()
from data_pre import DATASET
full_dataset = DATASET()
T=0
ACC=[]
SEN=[]
SPE=[]
BAC=[]
PPV=[]
NPV=[]
PRE=[]
REC=[]
F1_SCORE=[]
AUC=[]
for seed in [1,2,3,4,5]:
    print(seed)
    from sklearn.model_selection import KFold
    k = argv.k_fold
    kfold = KFold(n_splits=k, random_state=seed, shuffle=True)
    Acc2 = []
    Sen2 = []
    Spe2 = []
    Bac2 = []
    Ppv2 = []
    Npv2 = []
    Pre2 = []
    Rec2 = []
    F1_score2 = []
    Auc2 = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(full_dataset)):
        print('------------fold no---------{}----------------------'.format(fold))
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=argv.minibatch_size, sampler=train_subsampler)  # 16,64
        test_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=argv.minibatch_size, sampler=test_subsampler)

        def calculate_metric(gt, pred):
            pred[pred > 0.5] = 1
            pred[pred < 1] = 0
            confusion = confusion_matrix(gt, pred)
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
            acc = (TP + TN) / float(TP + TN + FP + FN)
            sen = TP / float(TP + FN)
            spe = TN / float(TN + FP)
            bac = (sen + spe) / 2
            ppv = TP / float(TP + FP)
            npv = TN / float(TN + FN)
            pre = TP / float(TP + FP)
            rec = TP / float(TP + FN)
            f1_score = 2 * pre * rec / (pre + rec)
            return acc, sen, spe, bac, ppv, npv, pre, rec, f1_score

        model = MDRL(
            input_dim=116,
            hidden_dim=argv.hidden_dim,
            num_classes=2,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout)

        def get_device():
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        device=get_device()
        print(device)
        model.to(device)
        Ep=argv.num_epochs
        Lr=argv.lr
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr= Lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=argv.max_lr, epochs=Ep,
                                                        steps_per_epoch=len(train_loader), pct_start=0.2,
                                                        div_factor=argv.max_lr /  Lr, final_div_factor=1000)

        for epoch in range(Ep):
            train_acc = 0.0
            train_loss = 0.0
            test_acc = 0.0
            test_loss = 0.0
            model.train()
            for i,(timeseries,label) in enumerate(train_loader):
                clip_grad=0.0
                dyn_a, sampling_points=util.bold.process_dynamic_fc(timeseries, argv.window_size, argv.window_stride, argv.dynamic_length)
                sampling_endpoints = [p+argv.window_size for p in sampling_points]
                dyn_v=torch.nan_to_num(dyn_a.float())
                t = timeseries.permute(1,0,2)
                label = label.long().to(device)
                t = t.to(torch.float32)
                optimizer.zero_grad()
                logit, reconstruct_loss, modularityloss,attention, latent, reg_ortho = model(dyn_v.to(device), dyn_a.to(device))
                pred = logit.argmax(1).to(device)
                prob = logit.softmax(1)
                batch_loss =  criterion(logit, label.to(device))+argv.lambda1*reconstruct_loss+argv.lambda2*modularityloss
                _, train_pred = torch.max(logit, 1)
                if optimizer is not None:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                train_acc += (train_pred.cpu() == label.cpu()).sum().item()
                train_loss += batch_loss.item()

            Labels = []
            Test_pred = []
            Pre_score = []
            model.eval()
            with torch.no_grad():
                for i,(timeseries,label) in enumerate(test_loader):
                    Labels.append(label)
                    dyn_a, sampling_points = util.bold.process_dynamic_fc(timeseries,argv.window_size, argv.window_stride, argv.dynamic_length)
                    sampling_endpoints = [p+argv.window_size for p in sampling_points]
                    dyn_v = torch.nan_to_num(dyn_a.float())
                    t = timeseries.permute(1, 0, 2)
                    label = label.long().to(device)
                    t = t.to(torch.float32)
                    logit, reconstruct_loss, modularityloss,attention, latent, reg_ortho = model(dyn_v.to(device), dyn_a.to(device))
                    batch_loss = criterion(logit, label.to(device))+argv.lambda1*reconstruct_loss+argv.lambda2*modularityloss
                    pre_socre = logit[:, 1]
                    Pre_score.append(pre_socre)
                    _, test_pred = torch.max(logit, 1)
                    Test_pred.append(test_pred)
                    test_acc += (
                            test_pred.cpu() == label.cpu()).sum().item()
                    test_loss += batch_loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Test Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, Ep, train_acc / len(train_idx), train_loss / len(train_loader),
                    test_acc / len(test_idx), test_loss / len(test_loader)
                ))
                y_true = torch.cat(Labels, -1).cpu()
                y_pred = torch.cat(Test_pred, -1).cpu()
                PPre_score = torch.cat(Pre_score, -1).cpu()
                acc, sen, spe, bac, ppv, npv, pre, rec, f1_score = calculate_metric(y_true, y_pred)
                from sklearn import metrics

                fpr, tpr, threshold = metrics.roc_curve(y_true, PPre_score)
                auc = metrics.auc(fpr, tpr)
        Acc2.append(acc)
        Sen2.append(sen)
        Spe2.append(spe)
        Bac2.append(bac)
        Ppv2.append(ppv)
        Npv2.append(npv)
        Pre2.append(pre)
        Rec2.append(rec)
        F1_score2.append(f1_score)
        Auc2.append(auc)
    avg_Acc = sum(Acc2) / k
    print(avg_Acc)
    print('Acc2std', np.std(Acc2, ddof=1))
    avg_Sen = sum(Sen2) / k
    print(avg_Sen)
    print('Sen2std', np.std(Sen2, ddof=1))
    avg_Spe = sum(Spe2) / k
    print(avg_Spe)
    print('Spe2std', np.std(Spe2, ddof=1))
    avg_Bac = sum(Bac2) / k
    print(avg_Bac)
    print('Bac2std', np.std(Bac2, ddof=1))
    avg_Ppv = sum(Ppv2) / k
    print(avg_Ppv)
    print('Ppv2std', np.std(Ppv2, ddof=1))
    avg_Npv = sum(Npv2) / k
    print(avg_Npv)
    print('Npv2std', np.std(Npv2, ddof=1))
    avg_Pre = sum(Pre2) / k
    print(avg_Pre)
    print('Pre2std', np.std(Pre2, ddof=1))
    avg_Rec = sum(Rec2) / k
    print(avg_Rec)
    print('Rec2std', np.std(Rec2, ddof=1))
    avg_F1_score = sum(F1_score2) / k
    print(avg_F1_score)
    print('F1_score2std', np.std(F1_score2, ddof=1))
    avg_Auc = sum(Auc2) / k
    print(avg_Auc)
    print('Auc2std', np.std(Auc2, ddof=1))
    ACC.extend([avg_Acc])
    SEN.extend([avg_Sen])
    SPE.extend([avg_Spe])
    BAC.extend([avg_Bac])
    PPV.extend([avg_Ppv])
    NPV.extend([avg_Npv])
    PRE.extend([avg_Pre])
    REC.extend([avg_Rec])
    F1_SCORE.extend([avg_F1_score])
    AUC.extend([avg_Auc])
print('ACCmean', np.mean(ACC))
print('ACCstd', np.std(ACC, ddof=1))

print('SENmean', np.mean(SEN))
print('SENstd', np.std(SEN, ddof=1))

print('SPEmean', np.mean(SPE))
print('SPEstd', np.std(SPE, ddof=1))

print('BACmean', np.mean(BAC))
print('BACstd', np.std(BAC, ddof=1))

print('PPVmean', np.mean(PPV))
print('PPVstd', np.std(PPV, ddof=1))

print('NPVmean', np.mean(NPV))
print('NPVstd', np.std(NPV, ddof=1))

print('PREmean', np.mean(PRE))
print('PREstd', np.std(PRE, ddof=1))

print('RECmean', np.mean(REC))
print('RECstd', np.std(REC, ddof=1))

print('F1_SCOREmean', np.mean(F1_SCORE))
print('F1_SCOREstd', np.std(F1_SCORE, ddof=1))

print('AUCmean', np.mean(AUC))
print('AUCstd', np.std(AUC, ddof=1))




