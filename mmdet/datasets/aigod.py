from .builder import DATASETS
from .coco import CocoDataset

import itertools
import logging
import random
import math
from collections import OrderedDict

import numpy as np
from mmcv.utils import print_log
from aigodpycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AIGODDataset(CocoDataset):

    CLASSES = ('group', )

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 classwise_lrp=True,
                 proposal_nums=(100, 300, 500),
                 iou_thrs=None,
                 metric_items=None,
                 apply_meanshift=False,
                 bandWidth=0.2,
                 with_lrp=False):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        if apply_meanshift:
            cluster_results = self.transfer_cluster(results, bandWidth)
        else:
            cluster_results = results

        result_files, tmp_dir = self.format_results(cluster_results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP@100': 0,
                'mAP@300': 1,
                'mAP@500': 2,
                'mAP_50': 3,
                'mAP_75': 4,
                'mAP_dim': 5,
                'mAP_tiny': 6,
                'mAP_small': 7,
                'mAP_medium': 8,
                'AR@100': 9,
                'AR@300': 10,
                'AR@500': 11,
                'AR_50': 12,
                'AR_75': 13,
                'AR_dim': 14,
                'AR_tiny': 15,
                'AR_small': 16,
                'AR_medium': 17,
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate(with_lrp=with_lrp)
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate(with_lrp=with_lrp)
                cocoEval.summarize()
                if classwise:
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                    # Compute per-category AR
                    recalls = cocoEval.eval['recall']
                    assert len(self.cat_ids) == recalls.shape[1]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        recall = recalls[:, idx, 0, -1]
                        recall = recall[recall > -1]
                        if recall.size:
                            ar = np.mean(recall)
                        else:
                            ar = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ar):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AR'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)
                
                if with_lrp and classwise_lrp:  
                    # Compute per-category oLRP
                    oLRPs = cocoEval.eval['olrp']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == oLRPs.shape[0]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        olrp = oLRPs[idx, 0, -1]
                        olrp = olrp[olrp > -1]
                        if olrp.size:
                            ap = np.mean(olrp)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'oLRP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP@100', 'mAP@300', 'mAP@500', 'mAP_50', 'mAP_75', 'mAP_dim', 'mAP_tiny', 'mAP_small', 'mAP_medium',
                        'AR@100', 'AR@300', 'AR@500', 'AR_50', 'AR_75', 'AR_dim', 'AR_tiny', 'AR_small', 'AR_medium',
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:9]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} {ap[4]:.3f}'
                    f'{ap[5]:.3f} {ap[6]:.3f} {ap[7]:.3f} {ap[8]:.3f}')
                ar = cocoEval.stats[9:]
                eval_results[f'{metric}_AR_copypaste'] = (
                    f'{ar[0]:.3f} {ar[1]:.3f} {ar[2]:.3f} {ar[3]:.3f} {ar[4]:.3f}'
                    f'{ar[5]:.3f} {ar[6]:.3f} {ar[7]:.3f} {ar[8]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def transfer_cluster(self, results, bandWidth):

        cluster_results = []
        for idx in range(len(results)):
            info = self.data_infos[idx]
            result = results[idx]
            cluster_result = []
            for label in range(len(result)):
                bboxes = result[label]

                if len(bboxes) == 0:
                    cluster_result.append(bboxes)
                    continue

                dataPts = (bboxes[:, :2] + bboxes[:, 2:4]) / 2

                width, height = info['width'], info['height']
                dataPts = dataPts / np.array([width, height])
                dataPts = dataPts.T

                clustCent, data2cluster = self.MeanShiftCluster(dataPts, bandWidth, plotFlag=False)

                if len(clustCent) == 0:
                    cluster_result.append(bboxes)
                    continue

                # get box of cluster
                cluster_num = clustCent.shape[1]
                gt_num_in_cluster = []
                for i in range(cluster_num):
                    gt_num = np.sum(data2cluster == i)
                    gt_num_in_cluster.append(gt_num)
                socred_ind = np.argsort(gt_num_in_cluster)[::-1]
                cluster_id_list = socred_ind

                cluster_box_list = []
                for i, clus_ind in enumerate(cluster_id_list[:cluster_num]):
                    
                    object_gts = bboxes[data2cluster == clus_ind]
                    if len(object_gts) == 0:
                        continue

                    # fuse object box to generate cluster box
                    x1 = object_gts[:, 0]
                    y1 = object_gts[:, 1]
                    x2 = object_gts[:, 2]
                    y2 = object_gts[:, 3]
                    s = object_gts[:, 4]
                    
                    x1 = np.min(x1)
                    y1 = np.min(y1)
                    x2 = np.max(x2)
                    y2 = np.max(y2)
                    s = np.mean(s)

                    cluster_box = [x1, y1, x2, y2, s]
                    cluster_box_list.append(cluster_box)
                
                cluster_box_list = np.array(cluster_box_list)
                cluster_result.append(cluster_box_list)
            cluster_results.append(cluster_result)
        
        return cluster_results
    
    def MeanShiftCluster(self, dataPts, bandWidth, plotFlag=False):
    
        # **** Initialize stuff ***
        numDim, numPts  = dataPts.shape
        numClust        = 0
        bandSq          = bandWidth**2
        initPtInds      = [ind for ind in range(numPts)]
        maxPos          = np.max(dataPts, axis=1)                           # biggest size in each dimension
        minPos          = np.min(dataPts, axis=1)                           # smallest size in each dimension
        boundBox        = maxPos - minPos;                                  # bounding box size
        stopThresh      = 1e-3*bandWidth                                    # when mean has converged
        clustCent       = []                                                # center of clust
        beenVisitedFlag = np.zeros((1,numPts), dtype=np.uint8)              # track if a points been seen already
        numInitPts      = numPts                                            # number of points to posibaly use as initilization points
        clusterVotes    = np.zeros((1,numPts), dtype=np.uint16)             # used to resolve conflicts on cluster membership
        
        while numInitPts:

            tempInd          = math.ceil((numInitPts - 1 - 1e-6) * random.random())        # pick a random seed point
            stInd            = initPtInds[tempInd]                                         # use this point as start of mean
            myMean           = np.expand_dims(dataPts[:, stInd], axis=1)                   # intilize mean to this points location
            myMembers        = []                                                          # points that will get added to this cluster                          
            thisClusterVotes = np.zeros((1,numPts), dtype=np.uint16)                       # used to resolve conflicts on cluster membership

            while True:

                sqDistToAll = np.sum((np.repeat(myMean, numPts, axis=1) - dataPts) ** 2, axis=0)    # dist squared from mean to all points still active
                inInds      = np.where(sqDistToAll < bandSq)                                        # points within bandWidth
                thisClusterVotes[:, inInds] = thisClusterVotes[:, inInds] + 1                       # add a vote for all the in points belonging to this cluster
                
                myOldMean = myMean;                                            # save the old mean
                myMean    = np.mean(dataPts[:, inInds], axis=2)                # compute the new mean
                myMembers = myMembers + inInds[0].tolist()                     # add any point within bandWidth to the cluster
                beenVisitedFlag[:, myMembers] = 1                              # mark that these points have been visited
            
                if plotFlag:
                    raise NotImplementedError("Have not implemented plot function")

                # **** if mean doesn't move much stop this cluster ***
                if np.linalg.norm(myMean - myOldMean) < stopThresh:

                    # check for merge posibilities
                    mergeWith = 0
                    for cN in range(numClust):
                        distToOther = np.linalg.norm(myMean - np.expand_dims(clustCent[:, cN], axis=1))     # distance from posible new clust max to old clust max
                        if distToOther < bandWidth / 2:                                                     # if its within bandwidth/2 merge new and old
                            mergeWith = cN
                            break

                    if mergeWith > 0:    # something to merge
                        clustCent[:, mergeWith]    = 0.5 * (myMean[:, 0] + clustCent[:, cN])           # record the max as the mean of the two merged (I know biased twoards new ones)
                        clusterVotes[mergeWith, :] = clusterVotes[mergeWith, :] + thisClusterVotes     # add these votes to the merged cluster
                    else:    #its a new cluster
                        numClust = numClust + 1                   # increment clusters
                        if len(clustCent) == 0:                   # record the mean  
                            clustCent = myMean
                        else:
                            clustCent = np.concatenate([clustCent, myMean], axis=1)    

                        if numClust == 1:             
                            clusterVotes[numClust - 1, :] = thisClusterVotes
                        else:
                            clusterVotes = np.concatenate([clusterVotes, thisClusterVotes], axis=0)
                    
                    break

            _, initPtInds = np.where(beenVisitedFlag == 0)         # we can initialize with any of the points not yet visited
            numInitPts    = len(initPtInds)                        # number of active points in set        
        
        val          = np.max(clusterVotes, axis=0)
        data2cluster = np.argmax(clusterVotes, axis=0)             # a point belongs to the cluster with the most votes
            
        return clustCent, data2cluster