from base.utils import Averager, set_gpu, ensure_path, get_dataloader, get_model, \
    LabelSmoothing, get_metrics, Timer
from base.supcon_loss import SupConLoss
import torch.nn as nn
import torch
import os.path as osp


CUDA = torch.cuda.is_available()


def train_one_epoch(data_loader, net, loss_fn, optimizer, supcon_loss_fn=None,
                    supcon_weight=0.0, use_mixup=False, mixup_alpha=0.4):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []

    use_supcon = supcon_loss_fn is not None and supcon_weight > 0

    for i, data_batch in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = data_batch[0].cuda(), data_batch[1].cuda()
        else:
            x_batch, y_batch = data_batch[0], data_batch[1]

        if x_batch.size(0) == 1:
            x_batch = torch.cat((x_batch, x_batch), dim=0)
            y_batch = torch.cat((y_batch, y_batch), dim=0)

        if use_mixup and hasattr(net, 'forward_mixup'):
            logits_mix, feat_mix, labels_a, labels_b, lam = net.forward_mixup(
                x_batch, y_batch, mixup_alpha)
            ce_loss = lam * loss_fn(logits_mix, labels_a) + (1 - lam) * loss_fn(logits_mix, labels_b)
            # Clean forward for SupCon and predictions
            out_clean, feat_clean = net(x_batch, return_feature=True)
            loss = ce_loss
            if use_supcon:
                loss = loss + supcon_weight * supcon_loss_fn(feat_clean, y_batch)
            _, pred = torch.max(out_clean, 1)
        elif use_supcon:
            out, feature = net(x_batch, return_feature=True)
            ce_loss = loss_fn(out, y_batch)
            sc_loss = supcon_loss_fn(feature, y_batch)
            loss = ce_loss + supcon_weight * sc_loss
            _, pred = torch.max(out, 1)
        else:
            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)

        tl.add(loss)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return tl.item(), pred_train, act_train


def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, data_batch in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = data_batch[0].cuda(), data_batch[1].cuda()
            else:
                x_batch, y_batch = data_batch[0], data_batch[1]

            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def set_up(args):
    if CUDA:
        set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, data_val, label_val, subject, trial):
    save_name = '_sub' + str(subject) + '_trial' + str(trial)
    set_up(args)

    train_loader = get_dataloader(data=data_train, label=label_train, batch_size=args.batch_size)
    val_loader = get_dataloader(data=data_val, label=label_val, batch_size=1)

    model = get_model(args)
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # SupCon loss setup
    supcon_weight = getattr(args, 'supcon_weight', 0.0)
    supcon_loss_fn = None
    if supcon_weight > 0:
        supcon_temp = getattr(args, 'supcon_temperature', 0.07)
        supcon_loss_fn = SupConLoss(temperature=supcon_temp)
        if CUDA:
            supcon_loss_fn = supcon_loss_fn.cuda()

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
        model_name_reproduce = 'sub' + str(subject) + '_trial' + str(trial)
        data_type = 'model_' + args.data_format + '_' + args.label_type
        ensure_path(osp.join(args.save_path, data_type))
        model_name_reproduce = osp.join(data_type, model_name_reproduce)
        torch.save(model.state_dict(), osp.join(args.save_path, model_name_reproduce + '.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()
    potential_epochs = [50, 100, 200, 400]
    MAX_EPOCH = 500
    model_saved = False

    for epoch in range(1, MAX_EPOCH + 1):
        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model,
            loss_fn=loss_fn, optimizer=optimizer,
            supcon_loss_fn=supcon_loss_fn, supcon_weight=supcon_weight,
            use_mixup=getattr(args, 'use_mixup', False),
            mixup_alpha=getattr(args, 'mixup_alpha', 0.4)
        )

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'.format(
            epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn)
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.format(
            epoch, loss_val, acc_val, f1_val))

        if acc_val >= trlog['max_acc'] and epoch >= int(0.2 * args.max_epoch) and acc_train >= 0.7:
            trlog['max_acc'] = acc_val
            save_model('candidate')
            print('Model saved!:{}'.format(acc_train))
            model_saved = True

        if model_saved and epoch >= args.max_epoch:
            print("Reach initial max epoch")
            break
        elif model_saved and epoch >= potential_epochs[0]:
            print("Reach max epoch: {}".format(potential_epochs[0]))
            break
        elif model_saved and epoch >= potential_epochs[1]:
            print("Reach max epoch: {}".format(potential_epochs[1]))
            break
        elif model_saved and epoch >= potential_epochs[2]:
            print("Reach max epoch: {}".format(potential_epochs[2]))
            break

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{} TRIAL:{}'.format(
            timer.measure(), timer.measure(epoch / args.max_epoch), subject, trial))

    assert model_saved, "No model is saved!!!"
    save_name_ = 'trlog' + save_name
    ensure_path(osp.join(args.save_path, 'log_train'))
    torch.save(trlog, osp.join(args.save_path, 'log_train', save_name_))

    return trlog['max_acc']


def test(args, data, label, reproduce, subject, trial, model_to_load='candidate.pth'):
    set_up(args)
    test_loader = get_dataloader(data=data, label=label, batch_size=1)

    model = get_model(args)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    if reproduce:
        model_name_reproduce = 'sub' + str(subject) + '_trial' + str(trial) + '.pth'
        data_type = 'model_' + args.data_format + '_' + args.label_type
        load_path_final = osp.join(args.save_path, data_type, model_name_reproduce)
        model.load_state_dict(torch.load(load_path_final))
    else:
        model.load_state_dict(torch.load(osp.join(args.save_path, model_to_load)))

    loss, pred, act = predict(data_loader=test_loader, net=model, loss_fn=loss_fn)
    acc, f1, _ = get_metrics(y_pred=pred, y_true=act)
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act
