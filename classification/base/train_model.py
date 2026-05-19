from base.utils import Averager, set_gpu, ensure_path, get_dataloader, get_model, \
    LabelSmoothing, get_metrics, Timer
from base.supcon_loss import SupConLoss
import torch.nn as nn
import torch.nn.functional as F
import torch
import os.path as osp


CUDA = torch.cuda.is_available()


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        self.reset(model)

    def reset(self, model):
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def update(self, model):
        state = model.state_dict()
        for key, value in state.items():
            value = value.detach()
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def clone_state_dict(model):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def prototype_margin_loss(features, labels, num_class, margin=0.2):
    features = F.normalize(features, dim=1)
    class_prototypes = []
    compact_loss = features.new_tensor(0.0)
    active_classes = 0

    for class_id in range(num_class):
        mask = labels == class_id
        if not torch.any(mask):
            continue
        prototype = F.normalize(features[mask].mean(dim=0, keepdim=True), dim=1)
        class_prototypes.append(prototype.squeeze(0))
        compact_loss = compact_loss + (1.0 - (features[mask] * prototype).sum(dim=1)).mean()
        active_classes += 1

    if active_classes == 0:
        return features.new_tensor(0.0)

    compact_loss = compact_loss / active_classes
    if len(class_prototypes) < 2:
        return compact_loss

    prototypes = torch.stack(class_prototypes)
    sim = torch.matmul(prototypes, prototypes.t())
    pair_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    separation_loss = F.relu(sim[pair_mask] + margin).mean()
    return compact_loss + separation_loss


def apply_calibration(logits, features=None, calibration=None):
    if calibration is None:
        return logits

    adjusted = logits
    temperature = float(calibration.get('temperature', 1.0))
    if temperature > 0:
        adjusted = adjusted / temperature

    prototypes = calibration.get('prototypes')
    prototype_weight = float(calibration.get('prototype_weight', 0.0))
    if prototypes is not None and features is not None and prototype_weight > 0:
        prototypes = prototypes.to(features.device)
        prototype_scale = float(calibration.get('prototype_scale', 1.0))
        proto_logits = torch.matmul(
            F.normalize(features, dim=1),
            F.normalize(prototypes, dim=1).t()
        ) * prototype_scale
        adjusted = adjusted + prototype_weight * proto_logits

    logit_bias = float(calibration.get('logit_bias', 0.0))
    if adjusted.size(1) == 2 and logit_bias != 0:
        adjusted = adjusted.clone()
        adjusted[:, 1] = adjusted[:, 1] + logit_bias

    return adjusted


def build_sapc_calibration(args, logits, features, labels):
    if not getattr(args, 'use_sapc', False):
        return None

    labels = labels.long()
    num_class = int(getattr(args, 'num_class', int(labels.max().item()) + 1))
    prototypes = []
    for class_id in range(num_class):
        mask = labels == class_id
        if torch.any(mask):
            prototypes.append(features[mask].mean(dim=0))
        else:
            prototypes.append(torch.zeros(features.size(1), dtype=features.dtype))
    prototypes = torch.stack(prototypes).cpu()

    weight_grid = [
        float(item.strip())
        for item in str(getattr(args, 'sapc_prototype_weights', '0,0.05,0.1,0.2,0.3')).split(',')
        if item.strip()
    ]
    if not weight_grid:
        weight_grid = [0.0]

    best = {
        'temperature': 1.0,
        'logit_bias': 0.0,
        'prototype_weight': 0.0,
        'prototype_scale': float(getattr(args, 'sapc_prototype_scale', 5.0)),
        'prototypes': prototypes,
        'score': -1.0,
        'acc': 0.0,
        'f1': 0.0,
    }

    with torch.no_grad():
        for weight in weight_grid:
            calibration = dict(best)
            calibration['prototype_weight'] = weight
            adjusted = apply_calibration(logits, features, calibration)

            if adjusted.size(1) == 2 and getattr(args, 'use_logit_calibration', False):
                margins = (adjusted[:, 1] - adjusted[:, 0]).detach().cpu()
                steps = int(getattr(args, 'calibration_steps', 121))
                thresholds = torch.linspace(
                    float(margins.min().item()) - 1.0,
                    float(margins.max().item()) + 1.0,
                    steps=max(3, steps)
                )
                biases = [-float(threshold.item()) for threshold in thresholds]
            else:
                biases = [0.0]

            for bias in biases:
                calibration['logit_bias'] = bias
                pred = torch.argmax(apply_calibration(logits, features, calibration), dim=1)
                acc, f1, _ = get_metrics(y_pred=pred.cpu().tolist(), y_true=labels.cpu().tolist())
                score = (
                    getattr(args, 'val_acc_weight', 0.5) * acc
                    + getattr(args, 'val_f1_weight', 0.5) * f1
                )
                if score > best['score']:
                    best = dict(calibration)
                    best['score'] = float(score)
                    best['acc'] = float(acc)
                    best['f1'] = float(f1)

    return best


def train_one_epoch(data_loader, net, loss_fn, optimizer, supcon_loss_fn=None,
                    supcon_weight=0.0, use_mixup=False, mixup_alpha=0.4,
                    prototype_weight=0.0, prototype_margin=0.2, num_class=2):
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
            if prototype_weight > 0:
                loss = loss + prototype_weight * prototype_margin_loss(
                    feat_clean, y_batch, num_class=num_class, margin=prototype_margin)
            _, pred = torch.max(out_clean, 1)
        elif use_supcon:
            out, feature = net(x_batch, return_feature=True)
            ce_loss = loss_fn(out, y_batch)
            sc_loss = supcon_loss_fn(feature, y_batch)
            loss = ce_loss + supcon_weight * sc_loss
            if prototype_weight > 0:
                loss = loss + prototype_weight * prototype_margin_loss(
                    feature, y_batch, num_class=num_class, margin=prototype_margin)
            _, pred = torch.max(out, 1)
        else:
            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)

        tl.add(loss.item())
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return tl.item(), pred_train, act_train


def predict(data_loader, net, loss_fn, calibration=None, return_extra=False):
    net.eval()
    pred_val = []
    act_val = []
    logits_all = []
    features_all = []
    vl = Averager()
    with torch.no_grad():
        for i, data_batch in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = data_batch[0].cuda(), data_batch[1].cuda()
            else:
                x_batch, y_batch = data_batch[0], data_batch[1]

            if calibration is not None or return_extra:
                out, feature = net(x_batch, return_feature=True)
            else:
                out = net(x_batch)
                feature = None
            loss = loss_fn(out, y_batch)
            out_for_pred = apply_calibration(out, feature, calibration)
            _, pred = torch.max(out_for_pred, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
            if return_extra:
                logits_all.append(out.detach().cpu())
                features_all.append(feature.detach().cpu())
    if return_extra:
        return vl.item(), pred_val, act_val, torch.cat(logits_all), torch.cat(features_all), torch.tensor(act_val)
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
    val_loader = get_dataloader(
        data=data_val, label=label_val, batch_size=args.batch_size, shuffle=False)

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

    def save_calibration(calibration):
        if calibration is None:
            return
        torch.save(calibration, osp.join(args.save_path, 'candidate_calibration.pth'))
        data_type = 'model_' + args.data_format + '_' + args.label_type
        ensure_path(osp.join(args.save_path, data_type))
        name = 'sub' + str(subject) + '_trial' + str(trial) + '_calibration.pth'
        torch.save(calibration, osp.join(args.save_path, data_type, name))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['val_f1'] = []
    trlog['max_acc'] = 0.0
    trlog['max_score'] = -1.0
    trlog['candidate_source'] = None
    trlog['candidate_calibration'] = None

    timer = Timer()
    potential_epochs = [50, 100, 200, 400]
    MAX_EPOCH = 500
    model_saved = False
    ema = ModelEMA(model, decay=getattr(args, 'ema_decay', 0.995)) if getattr(args, 'use_ema', False) else None
    ema_started = False
    ema_start_epoch = max(1, int(args.max_epoch * getattr(args, 'ema_start_ratio', 0.6)))

    for epoch in range(1, MAX_EPOCH + 1):
        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model,
            loss_fn=loss_fn, optimizer=optimizer,
            supcon_loss_fn=supcon_loss_fn, supcon_weight=supcon_weight,
            use_mixup=getattr(args, 'use_mixup', False),
            mixup_alpha=getattr(args, 'mixup_alpha', 0.4),
            prototype_weight=getattr(args, 'prototype_weight', 0.0),
            prototype_margin=getattr(args, 'prototype_margin', 0.2),
            num_class=getattr(args, 'num_class', 2)
        )

        if ema is not None and epoch >= ema_start_epoch:
            if not ema_started:
                ema.reset(model)
                ema_started = True
            else:
                ema.update(model)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'.format(
            epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val, logits_val, features_val, labels_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn, return_extra=True)
        calibration = build_sapc_calibration(args, logits_val, features_val, labels_val)
        if calibration is not None:
            _, pred_val, act_val = predict(
                data_loader=val_loader, net=model, loss_fn=loss_fn, calibration=calibration)
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.format(
            epoch, loss_val, acc_val, f1_val))

        val_score = (
            getattr(args, 'val_acc_weight', 0.5) * acc_val
            + getattr(args, 'val_f1_weight', 0.5) * f1_val
            - getattr(args, 'val_loss_weight', 0.0) * loss_val
        )
        candidate = {
            'score': val_score,
            'acc': acc_val,
            'f1': f1_val,
            'loss': loss_val,
            'source': 'raw',
            'calibration': calibration,
        }

        raw_state = None
        if ema is not None and ema_started:
            raw_state = clone_state_dict(model)
            ema.copy_to(model)
            loss_ema, pred_ema, act_ema, logits_ema, features_ema, labels_ema = predict(
                data_loader=val_loader, net=model, loss_fn=loss_fn, return_extra=True)
            calibration_ema = build_sapc_calibration(args, logits_ema, features_ema, labels_ema)
            if calibration_ema is not None:
                _, pred_ema, act_ema = predict(
                    data_loader=val_loader, net=model, loss_fn=loss_fn, calibration=calibration_ema)
            acc_ema, f1_ema, _ = get_metrics(y_pred=pred_ema, y_true=act_ema)
            score_ema = (
                getattr(args, 'val_acc_weight', 0.5) * acc_ema
                + getattr(args, 'val_f1_weight', 0.5) * f1_ema
                - getattr(args, 'val_loss_weight', 0.0) * loss_ema
            )
            if score_ema > candidate['score']:
                candidate = {
                    'score': score_ema,
                    'acc': acc_ema,
                    'f1': f1_ema,
                    'loss': loss_ema,
                    'source': 'ema',
                    'calibration': calibration_ema,
                }
            model.load_state_dict(raw_state, strict=True)

        if candidate['score'] >= trlog['max_score'] and epoch >= int(0.2 * args.max_epoch) and acc_train >= 0.7:
            trlog['max_score'] = candidate['score']
            trlog['max_acc'] = candidate['acc']
            trlog['candidate_source'] = candidate['source']
            trlog['candidate_calibration'] = {
                key: value for key, value in (candidate['calibration'] or {}).items()
                if key != 'prototypes'
            }
            if candidate['source'] == 'ema':
                ema.copy_to(model)
            save_model('candidate')
            save_calibration(candidate['calibration'])
            if raw_state is not None:
                model.load_state_dict(raw_state, strict=True)
            print('Model saved!:{}'.format(acc_train))
            print('Candidate source:{} score={:.4f} val_acc={:.4f} val_f1={:.4f}'.format(
                candidate['source'], candidate['score'], candidate['acc'], candidate['f1']))
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
        trlog['val_f1'].append(f1_val)

        print('ETA:{}/{} SUB:{} TRIAL:{}'.format(
            timer.measure(), timer.measure(epoch / args.max_epoch), subject, trial))

    assert model_saved, "No model is saved!!!"
    save_name_ = 'trlog' + save_name
    ensure_path(osp.join(args.save_path, 'log_train'))
    torch.save(trlog, osp.join(args.save_path, 'log_train', save_name_))

    return trlog['max_acc']


def test(args, data, label, reproduce, subject, trial, model_to_load='candidate.pth'):
    set_up(args)
    test_loader = get_dataloader(
        data=data, label=label, batch_size=args.batch_size, shuffle=False)

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

    calibration = None
    if getattr(args, 'use_sapc', False):
        if reproduce:
            data_type = 'model_' + args.data_format + '_' + args.label_type
            cal_name = 'sub' + str(subject) + '_trial' + str(trial) + '_calibration.pth'
            cal_path = osp.join(args.save_path, data_type, cal_name)
        else:
            cal_path = osp.join(args.save_path, 'candidate_calibration.pth')
        if osp.exists(cal_path):
            calibration = torch.load(cal_path, map_location='cpu', weights_only=False)

    loss, pred, act = predict(data_loader=test_loader, net=model, loss_fn=loss_fn, calibration=calibration)
    acc, f1, _ = get_metrics(y_pred=pred, y_true=act)
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act
