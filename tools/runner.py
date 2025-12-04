import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import matplotlib.pyplot as plt

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    (_, test_dataloader) = builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    start_epoch = 0
    best_metrics = None
    metrics = None

    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()

    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['TotalLoss', 'ShapeRecon', 'ShapeMatch', 'LatentLoss', 'NCCLoss'])


        num_iter = 0
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            dataset_name = config.dataset.train.NAME
            if dataset_name == 'EPN3DComplete' or dataset_name == 'PCNCompleteDataset':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'EPN3D':
                partial = data[0].cuda()
            elif dataset_name == 'ScanNet':
                partial = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1

            loss, loss_shape_recon, loss_shape_match, loss_latent, loss_ncc = base_model.module.get_loss(partial)
            loss.backward()

            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss.detach(), args)
                loss1 = dist_utils.reduce_tensor(loss1.detach(), args)
                loss2 = dist_utils.reduce_tensor(loss2.detach(), args)
                loss3 = dist_utils.reduce_tensor(loss3.detach(), args)
                latent_loss = dist_utils.reduce_tensor(latent_loss.detach(), args)
                ncc_loss = dist_utils.reduce_tensor(ncc_loss.detach(), args)
                losses.update([loss.item(), loss_shape_recon.item(), loss_shape_match.item(), loss_latent.item(), loss_ncc.item()])

            else:
                losses.update([loss.item(), loss_shape_recon.item(), loss_shape_match.item(), loss_latent.item(), loss_ncc.item()])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.6f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if (config.max_epoch - epoch) < 3:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)

# validate, test_net, and test remain unchanged


def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()

    test_losses = AverageMeter(['LossL1', 'LossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0]
            model_id = model_ids[0]

            partial = data[0].cuda()
            gt = data[1].cuda()

            coarse_points = base_model(partial)


            _metrics = Metrics.get(coarse_points, gt)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if (idx + 1) % 2400 == 0:
                print_log(f'Test[{idx + 1}/{len(test_dataloader)}] Taxonomy = {taxonomy_id} Sample = {model_id} Metrics = {["%.4f" % m for m in _metrics]}', logger=logger)

        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log(f'[Validation] EPOCH: {epoch} Metrics = {["%.4f" % m for m in test_metrics.avg()]}', logger=logger)

    return Metrics(config.consider_metric, test_metrics.avg())

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    if args.save_pred:
        print_log('Save Predictions', logger=logger)

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()
    from utils.metrics import Metrics
    category_metrics = dict()
    taxonomy_names = dict()

    if args.save_pred:
        pred_save_path = os.path.join(args.experiment_path, 'predictions')
        print_log(f"Saving path: {pred_save_path}", logger)
        if not os.path.exists(pred_save_path):
            os.makedirs(pred_save_path)
        from utils.o3d_misc import point_save

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0]
            model_id = model_ids[0]
            taxonomy_names[taxonomy_id] = config.dataset.test.others.class_choice[0]

            partial = data[0].cuda()
            gt = data[1].cuda()

            pred = base_model(partial)

            # Save predictions
            if args.save_pred:
                for b in range(pred.shape[0]):
                    sample_id = f'{idx * pred.shape[0] + b:05d}_{model_ids[b]}'
                    point_save(partial[b].cpu(), pred_save_path, f'{sample_id}_input', type='ply')
                    point_save(pred[b].cpu(), pred_save_path, f'{sample_id}_pred', type='ply')
                    point_save(gt[b].cpu(), pred_save_path, f'{sample_id}_gt', type='ply')

            _metrics = Metrics.get(pred, gt)
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = {'metrics_sum': [0.0, 0.0, 0.0], 'count': 0}
            for i in range(3):
                category_metrics[taxonomy_id]['metrics_sum'][i] += _metrics[i]
            category_metrics[taxonomy_id]['count'] += 1

            for b in range(pred.shape[0]):
                sample_idx = idx * pred.shape[0] + b + 1
                model_name = model_ids[b]
                metrics_str = [f"{m:.4f}" for m in _metrics]
                if sample_idx % 200 == 0 or sample_idx == len(test_dataloader.dataset):
                    print_log(f"Test[{sample_idx}/{len(test_dataloader.dataset)}] "
                              f"Taxonomy = {taxonomy_id} Sample = {model_name} "
                              f"Metrics = {metrics_str}", logger)


    # Final formatted output
    print_log("="*75, logger)
    print_log("Taxonomy\t#Sample\tCDL1\tCDL2\tUCDL2\t#ModelName", logger)

    overall_metrics = [0.0, 0.0, 0.0]
    total_samples = 0

    for taxonomy_id, value in category_metrics.items():
        count = value['count']
        mean_metrics = [m / count for m in value['metrics_sum']]
        for i in range(3):
            overall_metrics[i] += mean_metrics[i] * count
        total_samples += count

        print_log(f"{taxonomy_id}\t{count}\t"
                  f"{mean_metrics[0]:.3f} \t{mean_metrics[1]:.3f} \t{mean_metrics[2]:.3f} \t"
                  f"{taxonomy_names[taxonomy_id]}", logger)

    if total_samples > 0:
        avg_metrics = [m / total_samples for m in overall_metrics]
    else:
        avg_metrics = [0.0, 0.0, 0.0]

    print_log(f"Overall \t{total_samples}\t"
              f"{avg_metrics[0]:.3f} \t{avg_metrics[1]:.3f} \t{avg_metrics[2]:.3f} \t", logger)

    return Metrics(config.consider_metric, avg_metrics)




