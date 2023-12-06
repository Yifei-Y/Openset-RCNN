import numpy as np
import datetime
import time
import copy
from collections import defaultdict
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


class OpensetCOCOEval(COCOeval):
    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        k_gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        unk_gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=1000))
        k_dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        unk_dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=1000))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(k_gts, self.cocoGt)
            _toMask(unk_gts, self.cocoGt)
            _toMask(k_dts, self.cocoDt)
            _toMask(unk_dts, self.cocoDt)
        # set ignore flag
        for kgt in k_gts:
            kgt['ignore'] = kgt['ignore'] if 'ignore' in kgt else 0
            kgt['ignore'] = 'iscrowd' in kgt and kgt['iscrowd']
        for ugt in unk_gts:
            ugt['ignore'] = ugt['ignore'] if 'ignore' in ugt else 0
            ugt['ignore'] = 'iscrowd' in ugt and ugt['iscrowd']
        self._k_gts = defaultdict(list)       # gt for evaluation
        self._ok_gts = defaultdict(list)
        self._unk_gts = defaultdict(list)
        self._k_dts = defaultdict(list)       # dt for evaluation
        self._unk_dts = defaultdict(list)
        for kgt in k_gts:
            self._k_gts[kgt['image_id'], kgt['category_id']].append(kgt)
        for cId in p.catIds:
            for kgt in k_gts:
                if kgt['category_id'] != cId:
                    self._ok_gts[kgt['image_id'], cId].append(kgt)
        for ugt in unk_gts:
            self._unk_gts[ugt['image_id']].append(ugt)
        for kdt in k_dts:
            self._k_dts[kdt['image_id'], kdt['category_id']].append(kdt)
        for udt in unk_dts:
            self._unk_dts[udt['image_id']].append(udt)
        self.evalImgs_kdt = defaultdict(list)   # per-image per-category evaluation results
        self.evalImgs_unkdt = defaultdict(list)
        self.eval_kdt = {}                  # accumulated evaluation results
        self.eval_unkdt = {}

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        self.ious_kdt_kgt = {(imgId, catId): self.computeIoU_kdt_kgt(imgId, catId) \
                            for imgId in p.imgIds
                            for catId in catIds}
        self.ious_kdt_okgt = {(imgId, catId): self.computeIoU_kdt_okgt(imgId, catId) \
                            for imgId in p.imgIds
                            for catId in catIds}
        self.ious_kdt_unkgt = {(imgId, catId): self.computeIoU_kdt_unkgt(imgId, catId) \
                            for imgId in p.imgIds
                            for catId in catIds}
        self.ious_unkdt_kgt = {(imgId): self.computeIoU_unkdt_kgt(imgId) for imgId in p.imgIds}
        self.ious_unkdt_unkgt = {(imgId): self.computeIoU_unkdt_unkgt(imgId) for imgId in p.imgIds}
        
        maxDet = p.maxDets[-1]
        self.evalImgs_kdt = [self.evaluateImg_kdt(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self.evalImgs_unkdt = [self.evaluateImg_unkdt(imgId, areaRng, maxDet)
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
        ]
        
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU_kdt_kgt(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._k_gts[imgId,catId]
            dt = self._k_dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._k_gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._k_dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious
    
    def computeIoU_kdt_okgt(self, imgId, catId):
        p = self.params
        gt = self._ok_gts[imgId, catId]
        dt = self._k_dts[imgId,catId]
        
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious
    
    def computeIoU_kdt_unkgt(self, imgId, catId):
        p = self.params
        gt = self._unk_gts[imgId]
        dt = self._k_dts[imgId,catId]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious
    
    def computeIoU_unkdt_kgt(self, imgId):
        p = self.params
        gt = [_ for cId in p.catIds for _ in self._k_gts[imgId,cId]]
        dt = self._unk_dts[imgId]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious
    
    def computeIoU_unkdt_unkgt(self, imgId):
        p = self.params
        gt = self._unk_gts[imgId]
        dt = self._unk_dts[imgId]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def evaluateImg_kdt(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params

        k_gt = self._k_gts[imgId,catId]
        ok_gt = self._ok_gts[imgId,catId]
        unk_gt = self._unk_gts[imgId]
        k_dt = self._k_dts[imgId,catId]

        for kg in k_gt:
            if kg['ignore'] or (kg['area']<aRng[0] or kg['area']>aRng[1]):
                kg['_ignore'] = 1
            else:
                kg['_ignore'] = 0
        for okg in ok_gt:
            if okg['ignore'] or (okg['area']<aRng[0] or okg['area']>aRng[1]):
                okg['_ignore'] = 1
            else:
                okg['_ignore'] = 0
        for ug in unk_gt:
            if ug['ignore'] or (ug['area']<aRng[0] or ug['area']>aRng[1]):
                ug['_ignore'] = 1
            else:
                ug['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        k_gtind = np.argsort([kg['_ignore'] for kg in k_gt], kind='mergesort')
        k_gt = [k_gt[i] for i in k_gtind]
        ok_gtind = np.argsort([okg['_ignore'] for okg in ok_gt], kind='mergesort')
        ok_gt = [ok_gt[i] for i in ok_gtind]
        unk_gtind = np.argsort([ug['_ignore'] for ug in unk_gt], kind='mergesort')
        unk_gt = [unk_gt[i] for i in unk_gtind]
        k_dtind = np.argsort([-kd['score'] for kd in k_dt], kind='mergesort')
        k_dt = [k_dt[i] for i in k_dtind[0:maxDet]]
        k_iscrowd = [int(o['iscrowd']) for o in k_gt]
        ok_iscrowd = [int(o['iscrowd']) for o in ok_gt]
        unk_iscrowd = [int(o['iscrowd']) for o in unk_gt]
        # load computed ious
        ious_kgt = (
            self.ious_kdt_kgt[imgId, catId][:, k_gtind] \
            if len(self.ious_kdt_kgt[imgId, catId]) > 0 else self.ious_kdt_kgt[imgId, catId]
        )
        ious_okgt = (
            self.ious_kdt_okgt[imgId, catId][:, ok_gtind] \
            if len(self.ious_kdt_okgt[imgId, catId]) > 0 else self.ious_kdt_okgt[imgId, catId]
        )
        ious_unkgt = (
            self.ious_kdt_unkgt[imgId, catId][:, unk_gtind] \
            if len(self.ious_kdt_unkgt[imgId, catId]) > 0 else self.ious_kdt_unkgt[imgId, catId]
        )

        T = len(p.iouThrs)
        KG = len(k_gt)
        OKG = len(ok_gt)
        UG = len(unk_gt)
        KD = len(k_dt)
        kgtm  = np.zeros((T,KG))
        okgtm = np.zeros((T,OKG))
        unkgtm = np.zeros((T,UG))
        kdtm_kgt  = np.zeros((T,KD))
        kdtm_okgt = np.zeros((T,KD))
        kdtm_unkgt  = np.zeros((T,KD))
        kgtIg = np.array([kg['_ignore'] for kg in k_gt])
        okgtIg = np.array([okg['_ignore'] for okg in ok_gt])
        unkgtIg = np.array([ug['_ignore'] for ug in unk_gt])
        kdtIg_kgt = np.zeros((T,KD))
        kdtIg_okgt = np.zeros((T,KD))
        kdtIg_unkgt = np.zeros((T,KD))

        if not len(ious_kgt)==0:
            for tind, t in enumerate(p.iouThrs):
                for kdind, kd in enumerate(k_dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for kgind, kg in enumerate(k_gt):
                        # if this gt already matched, and not a crowd, continue
                        if kgtm[tind,kgind]>0 and not k_iscrowd[kgind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and kgtIg[m]==0 and kgtIg[kgind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious_kgt[kdind,kgind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious_kgt[kdind,kgind]
                        m=kgind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    kdtIg_kgt[tind,kdind] = kgtIg[m]
                    kdtm_kgt[tind,kdind]  = k_gt[m]['id']
                    kgtm[tind,m] = kd['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([kd['area']<aRng[0] or kd['area']>aRng[1] for kd in k_dt]).reshape((1, len(k_dt)))
        kdtIg_kgt = np.logical_or(kdtIg_kgt, np.logical_and(kdtm_kgt==0, np.repeat(a,T,0)))

        if not len(ious_okgt)==0:
            for tind, t in enumerate(p.iouThrs):
                for kdind, kd in enumerate(k_dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for okgind, okg in enumerate(ok_gt):
                        # if this gt already matched, and not a crowd, continue
                        if okgtm[tind,okgind]>0 and not ok_iscrowd[okgind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and okgtIg[m]==0 and okgtIg[okgind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious_okgt[kdind,okgind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious_okgt[kdind,okgind]
                        m=okgind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    kdtIg_okgt[tind,kdind] = okgtIg[m]
                    kdtm_okgt[tind,kdind]  = ok_gt[m]['id']
                    okgtm[tind,m]     = kd['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([kd['area']<aRng[0] or kd['area']>aRng[1] for kd in k_dt]).reshape((1, len(k_dt)))
        kdtIg_okgt = np.logical_or(kdtIg_okgt, np.logical_and(kdtm_okgt==0, np.repeat(a,T,0)))

        if not len(ious_unkgt)==0:
            for tind, t in enumerate(p.iouThrs):
                for kdind, kd in enumerate(k_dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for unkgind, unkg in enumerate(unk_gt):
                        # if this gt already matched, and not a crowd, continue
                        if unkgtm[tind,unkgind]>0 and not unk_iscrowd[unkgind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and unkgtIg[m]==0 and unkgtIg[unkgind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious_unkgt[kdind,unkgind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious_unkgt[kdind,unkgind]
                        m=unkgind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    kdtIg_unkgt[tind,kdind] = unkgtIg[m]
                    kdtm_unkgt[tind,kdind]  = unk_gt[m]['id']
                    unkgtm[tind,m]     = kd['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([kd['area']<aRng[0] or kd['area']>aRng[1] for kd in k_dt]).reshape((1, len(k_dt)))
        kdtIg_unkgt = np.logical_or(kdtIg_unkgt, np.logical_and(kdtm_unkgt==0, np.repeat(a,T,0)))

        # store results for given image and category
        return {
                'image_id':            imgId,
                'category_id':         catId,
                'aRng':                aRng,
                'maxDet':              maxDet,
                'kdtIds':              [kd['id'] for kd in k_dt],
                'kgtIds':              [kg['id'] for kg in k_gt],
                'okgtIds':             [okg['id'] for okg in ok_gt],
                'unkgtIds':            [ug['id'] for ug in unk_gt],
                'Matches_kdt_kgt':     kdtm_kgt,
                'Matches_kdt_okgt':    kdtm_okgt,
                'Matches_kdt_unkgt':   kdtm_unkgt,
                'kgtMatches':          kgtm,
                'okgtMatches':         okgtm,
                'unkgtMatches':        unkgtm,
                'kdtScores':           [kd['score'] for kd in k_dt],
                'kgtIgnore':           kgtIg,
                'okgtIgnore':          okgtIg,
                'unkgtIgnore':         unkgtIg,
                'kdtIgnore_kgt':       kdtIg_kgt,
                'kdtIgnore_okgt':      kdtIg_okgt,
                'kdtIgnore_unkgt':     kdtIg_unkgt,
            }
    
    def evaluateImg_unkdt(self, imgId, aRng, maxDet):
        '''
        '''
        p = self.params
        k_gt = [_ for cId in p.catIds for _ in self._k_gts[imgId,cId]]
        unk_gt = self._unk_gts[imgId]
        unk_dt = self._unk_dts[imgId]
        if len(unk_gt) == 0 and len(unk_dt) == 0:
            return None
        
        for kg in k_gt:
            if kg['ignore'] or (kg['area']<aRng[0] or kg['area']>aRng[1]):
                kg['_ignore'] = 1
            else:
                kg['_ignore'] = 0
        for ug in unk_gt:
            if ug['ignore'] or (ug['area']<aRng[0] or ug['area']>aRng[1]):
                ug['_ignore'] = 1
            else:
                ug['_ignore'] = 0
        
        # sort dt highest score first, sort gt ignore last
        kgtind = np.argsort([kg['_ignore'] for kg in k_gt], kind='mergesort')
        k_gt = [k_gt[i] for i in kgtind]
        unk_gtind = np.argsort([ug['_ignore'] for ug in unk_gt], kind='mergesort')
        unk_gt = [unk_gt[i] for i in unk_gtind]
        udtind = np.argsort([-ud['score'] for ud in unk_dt], kind='mergesort')
        unk_dt = [unk_dt[i] for i in udtind[0:maxDet]]
        k_iscrowd = [int(o['iscrowd']) for o in k_gt]
        unk_iscrowd = [int(o['iscrowd']) for o in unk_gt]

        # load computed ious
        ious_kgt = (
            self.ious_unkdt_kgt[imgId][:, kgtind] \
            if len(self.ious_unkdt_kgt[imgId]) > 0 else self.ious_unkdt_kgt[imgId]
        )
        ious_unkgt = (
            self.ious_unkdt_unkgt[imgId][:, unk_gtind] \
            if len(self.ious_unkdt_unkgt[imgId]) > 0 else self.ious_unkdt_unkgt[imgId]
        )

        T = len(p.iouThrs)
        KG = len(k_gt)
        UG = len(unk_gt)
        UD = len(unk_dt)
        kgtm  = np.zeros((T,KG))
        unkgtm = np.zeros((T,UG))
        unkdtm_kgt  = np.zeros((T,UD))
        unkdtm_unkgt  = np.zeros((T,UD))
        kgtIg = np.array([g['_ignore'] for g in k_gt])
        unkgtIg = np.array([ug['_ignore'] for ug in unk_gt])
        unkdtIg_kgt = np.zeros((T,UD))
        unkdtIg_unkgt = np.zeros((T,UD))

        if not len(ious_kgt)==0:
            for tind, t in enumerate(p.iouThrs):
                for udind, ud in enumerate(unk_dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for kgind, kg in enumerate(k_gt):
                        # if this gt already matched, and not a crowd, continue
                        if kgtm[tind,kgind]>0 and not k_iscrowd[kgind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and kgtIg[m]==0 and kgtIg[kgind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious_kgt[udind,kgind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious_kgt[udind,kgind]
                        m=kgind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    unkdtIg_kgt[tind,udind] = kgtIg[m]
                    unkdtm_kgt[tind,udind]  = k_gt[m]['id']
                    kgtm[tind,m]     = ud['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([ud['area']<aRng[0] or ud['area']>aRng[1] for ud in unk_dt]).reshape((1, len(unk_dt)))
        unkdtIg_kgt = np.logical_or(unkdtIg_kgt, np.logical_and(unkdtm_kgt==0, np.repeat(a,T,0)))

        if not len(ious_unkgt)==0:
            for tind, t in enumerate(p.iouThrs):
                for udind, ud in enumerate(unk_dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for unkgind, unkg in enumerate(unk_gt):
                        # if this gt already matched, and not a crowd, continue
                        if unkgtm[tind,unkgind]>0 and not unk_iscrowd[unkgind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and unkgtIg[m]==0 and unkgtIg[unkgind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious_unkgt[udind,unkgind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious_unkgt[udind,unkgind]
                        m=unkgind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    unkdtIg_unkgt[tind,udind] = unkgtIg[m]
                    unkdtm_unkgt[tind,udind]  = unk_gt[m]['id']
                    unkgtm[tind,m]     = ud['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([ud['area']<aRng[0] or ud['area']>aRng[1] for ud in unk_dt]).reshape((1, len(unk_dt)))
        unkdtIg_unkgt = np.logical_or(unkdtIg_unkgt, np.logical_and(unkdtm_unkgt==0, np.repeat(a,T,0)))

        # store results for given image and category
        return {
                'image_id':           imgId,
                'aRng':                aRng,
                'maxDet':              maxDet,
                'unkdtIds':            [ud['id'] for ud in unk_dt],
                'kgtIds':              [kg['id'] for kg in k_gt],
                'unkgtIds':            [ug['id'] for ug in unk_gt],
                'Matches_unkdt_kgt':   unkdtm_kgt,
                'Matches_unkdt_unkgt': unkdtm_unkgt,
                'kgtMatches':          kgtm,
                'unkgtMatches':        unkgtm,
                'unkdtScores':         [ud['score'] for ud in unk_dt],
                'kgtIgnore':           kgtIg,
                'unkgtIgnore':         unkgtIg,
                'unkdtIgnore_kgt':     unkdtIg_kgt,
                'unkdtIgnore_unkgt':   unkdtIg_unkgt,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results of known detections...')
        tic = time.time()
        if not self.evalImgs_kdt or not self.evalImgs_unkdt:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))
        ok_det_as_known = np.zeros((T,K,A,M))
        unk_det_as_known = np.zeros((T,K,A,M))
        fp_os = np.zeros((T,R,K,A,M))
        tp_plus_fp_cs = np.zeros((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs_kdt[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['kdtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]
                    
                    dtScoresSortedExpand = np.expand_dims(dtScoresSorted, 0)
                    dtScoresSortedExpand = np.repeat(dtScoresSortedExpand, T, 0)
                    kdtm_kgt  = np.concatenate([e['Matches_kdt_kgt'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    kdtm_okgt = np.concatenate([e['Matches_kdt_okgt'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    kdtm_unkgt = np.concatenate([e['Matches_kdt_unkgt'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    kdtIg_kgt = np.concatenate([e['kdtIgnore_kgt'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    kdtIg_okgt = np.concatenate([e['kdtIgnore_okgt'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    kdtIg_unkgt = np.concatenate([e['kdtIgnore_unkgt'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    kgtIg = np.concatenate([e['kgtIgnore'] for e in E])
                    npig = np.count_nonzero(kgtIg==0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(kdtm_kgt, np.logical_not(kdtIg_kgt) )
                    fps = np.logical_and(np.logical_not(kdtm_kgt), np.logical_not(kdtIg_kgt) )
                    okfps = np.logical_and(kdtm_okgt, np.logical_not(kdtIg_okgt))
                    ufps = np.logical_and(kdtm_unkgt, np.logical_not(kdtIg_unkgt))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    tp_fp_sum = tp_sum + fp_sum
                    okfp_sum = np.sum(okfps, axis=1).astype(dtype=np.float)
                    ufp_sum = np.cumsum(ufps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp, tp_fp, ufp) in enumerate(zip(tp_sum, fp_sum, tp_fp_sum, ufp_sum)):
                        if len(ufp):
                            unk_det_as_known[t,k,a,m] = ufp[-1]

                        ok_det_as_known[t,k,a,m] = okfp_sum[t]

                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))
                        tf = np.zeros((R,))
                        fo = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        l = len(tp_fp)
                        if l:
                            for ri, pi in enumerate(inds):
                                if pi == l:
                                    pi -= 1
                                tf[ri] = tp_fp[pi]
                                fo[ri] = ufp[pi]
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
                        tp_plus_fp_cs[t,:,k,a,m] = np.array(tf)
                        fp_os[t,:,k,a,m] = np.array(fo)
        self.eval_kdt = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
            'ok_det_as_known': ok_det_as_known,
            'unk_det_as_known': unk_det_as_known,
            'tp_plus_fp_cs': tp_plus_fp_cs,
            'fp_os': fp_os
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

        print('Accumulating evaluation results of unknown detections...')
        tic = time.time()
        if not self.evalImgs_unkdt:
            print('Please run evaluate() first')
        
        precision   = -np.ones((T,R,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,A,M))
        scores      = -np.ones((T,R,A,M))

        num_k_det_as_unk = np.zeros((T,A,M))

        # retrieve E at each category, area range, and max number of detections
        for a, a0 in enumerate(a_list):
            Na = a0*I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs_unkdt[Na + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                udtScores = np.concatenate([e['unkdtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-udtScores, kind='mergesort')
                udtScoresSorted = udtScores[inds]

                udtm_kgt = np.concatenate([e['Matches_unkdt_kgt'][:,0:maxDet] for e in E], axis=1)[:,inds]
                udtm_unkgt = np.concatenate([e['Matches_unkdt_unkgt'][:,0:maxDet] for e in E], axis=1)[:,inds]
                udtIg_kgt = np.concatenate([e['unkdtIgnore_kgt'][:,0:maxDet] for e in E], axis=1)[:,inds]
                udtIg_unkgt = np.concatenate([e['unkdtIgnore_unkgt'][:,0:maxDet] for e in E], axis=1)[:,inds]
                kgtIg = np.concatenate([e['kgtIgnore'] for e in E])
                unkgtIg = np.concatenate([e['unkgtIgnore'] for e in E])
                npig = np.count_nonzero(unkgtIg==0 )
                if npig == 0:
                    continue

                tps = np.logical_and(udtm_unkgt, np.logical_not(udtIg_unkgt) )
                fps = np.logical_and(np.logical_not(udtm_unkgt), np.logical_not(udtIg_unkgt) )
                k_det_as_unk_fps = np.logical_and(udtm_kgt, np.logical_not(udtIg_kgt))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                k_det_as_unk_fp_sum = np.cumsum(k_det_as_unk_fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp, k_det_as_unk_fp) in enumerate(zip(tp_sum, fp_sum, k_det_as_unk_fp_sum)):
                    if len(k_det_as_unk_fp):
                        num_k_det_as_unk[t,a,m] = k_det_as_unk_fp[-1]
                    
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp+tp+np.spacing(1))
                    q  = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t,a,m] = rc[-1]
                    else:
                        recall[t,a,m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist(); q = q.tolist()

                    for i in range(nd-1, 0, -1):
                        if pr[i] > pr[i-1]:
                            pr[i-1] = pr[i]

                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = udtScoresSorted[pi]
                    except:
                        pass
                    precision[t,:,a,m] = np.array(q)
                    scores[t,:,a,m] = np.array(ss)
        
        self.eval_unkdt = {
            'params': p,
            'counts': [T, R, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
            'k_det_as_unk': num_k_det_as_unk
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _num_unk_det_as_known(iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {}'
            titleStr = 'UNK_det_as_K'
            typeStr = '(AOSE)'
            iouStr = '{:0.2f}'.format(iouThr)
            tind = [i for i, iouT in enumerate(p.iouThrs) if iouT == iouThr]
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            unk_det_as_known = self.eval_kdt['unk_det_as_known']

            self.unk_det_as_known = unk_det_as_known[tind,:,aind,mind]

            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, np.sum(unk_det_as_known[tind,:,aind,mind])))
            print(unk_det_as_known[tind,:,aind,mind])
            
            return np.sum(unk_det_as_known[tind,:,aind,mind])

        def _num_k_det_as_unk(iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {}'
            titleStr = 'K_det_as_UNK'
            iouStr = '{:0.2f}'.format(iouThr)
            tind = [i for i, iouT in enumerate(p.iouThrs) if iouT == iouThr]
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            k_det_as_unk = self.eval_unkdt['k_det_as_unk']

            self.k_det_as_unk = k_det_as_unk[tind,aind,mind]

            print(iStr.format(titleStr, iouStr, areaRng, maxDets, k_det_as_unk[tind,aind,mind]))
            
            return k_det_as_unk[tind,aind,mind]
        
        def _wi(iouThr=None, areaRng='all', maxDets=100, recall_level=0.8):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Wilderness Impact'
            typeStr = '(WI)'
            iouStr = '{:0.2f}'.format(iouThr)

            tind = [i for i, iouT in enumerate(p.iouThrs) if iouT == iouThr]
            rind = [i for i, recT in enumerate(p.recThrs) if recT == recall_level]
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            tp_plus_fp_cs = self.eval_kdt['tp_plus_fp_cs']
            fp_os = self.eval_kdt['fp_os']

            wi = np.mean(fp_os[tind,rind,:,aind,mind]) / np.mean(tp_plus_fp_cs[tind,rind,:,aind,mind])
            
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, wi))

            return wi
        
        def _print_precision(iouThr=.5, areaRng='all', maxDets=100 ):
            p = self.params

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # dimension of precision: [TxRxKxAxM]
            s = self.eval_kdt['precision']
            # IoU

            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
            s = np.squeeze(s[:,:,:,aind,mind])
            s = s[[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],:]
            
            for i in range(s.shape[1]):
                print(s[:,i])

        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Known Average Precision' if ap == 1 else 'Known Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval_kdt['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval_kdt['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarize_unk( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Unknown Average Precision' if ap == 1 else 'Unknown Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval_unkdt['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval_unkdt['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((30,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[-1])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[-1])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[-1])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[-1])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[-1])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, maxDets=self.params.maxDets[3])
            stats[10] = _summarize(0, maxDets=self.params.maxDets[4])
            stats[11] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[-1])
            stats[12] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[-1])
            stats[13] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[-1])
            stats[14] = _wi(iouThr=.5, areaRng='all', maxDets=100, recall_level=0.8)
            stats[15] = _num_unk_det_as_known(iouThr=.5, areaRng='all', maxDets=100)
            
            stats[16] = _summarize_unk(1)
            stats[17] = _summarize_unk(1, iouThr=.5, maxDets=self.params.maxDets[-1])
            stats[18] = _summarize_unk(1, iouThr=.75, maxDets=self.params.maxDets[-1])
            stats[19] = _summarize_unk(1, areaRng='small', maxDets=self.params.maxDets[-1])
            stats[20] = _summarize_unk(1, areaRng='medium', maxDets=self.params.maxDets[-1])
            stats[21] = _summarize_unk(1, areaRng='large', maxDets=self.params.maxDets[-1])
            stats[22] = _summarize_unk(0, maxDets=self.params.maxDets[0])
            stats[23] = _summarize_unk(0, maxDets=self.params.maxDets[1])
            stats[24] = _summarize_unk(0, maxDets=self.params.maxDets[2])
            stats[25] = _summarize_unk(0, maxDets=self.params.maxDets[3])
            stats[26] = _summarize_unk(0, maxDets=self.params.maxDets[4])
            stats[27] = _summarize_unk(0, areaRng='small', maxDets=self.params.maxDets[-1])
            stats[28] = _summarize_unk(0, areaRng='medium', maxDets=self.params.maxDets[-1])
            stats[29] = _summarize_unk(0, areaRng='large', maxDets=self.params.maxDets[-1])
            return stats
        
        if not self.eval_kdt or not self.eval_unkdt:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        self.stats = summarize()