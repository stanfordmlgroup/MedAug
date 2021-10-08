# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import warnings
import random


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07,
                 mlp=False, pretrained=False,
                 positive_strat={
                     "same_study": True, "diff_study": False,
                     "same_lat": False, "diff_lat": False},
                 negative_strat={
                     "hard_neg": 'random', "target_hard_neg": None,
                     "subsample": None, 'append_neg': None,
                     "syn_hard": None}):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.positive_strat = positive_strat

        self.hard_negative = negative_strat['hard_neg']
        self.target_hard_negative = negative_strat['target_hard_neg']
        self.subsample = negative_strat['subsample']
        self.append_hard_negative = negative_strat['append_neg']
        self.synthesize_hard_samples = negative_strat['syn_hard']
        self.negative_laterality_only = negative_strat['neg_lat_only']

        if self.synthesize_hard_samples:
            assert self.append_hard_negative > 0

        # create the encoders
        # num_classes is the output fc dimension
        if pretrained:

            self.encoder_q = base_encoder(pretrained=True)
            self.encoder_k = base_encoder(pretrained=True)
            if self.encoder_q.__class__.__name__.lower() == 'resnet':
                num_ftrs_q = self.encoder_q.fc.in_features
                num_ftrs_k = self.encoder_k.fc.in_features
                self.encoder_q.fc = nn.Linear(num_ftrs_q, dim)
                self.encoder_k.fc = nn.Linear(num_ftrs_k, dim)
            elif self.encoder_q.__class__.__name__.lower() == 'mnasnet':
                num_ftrs_q = self.encoder_q.classifier._modules['1'].in_features
                num_ftrs_k = self.encoder_k.classifier._modules['1'].in_features
                self.encoder_q.classifier = nn.Linear(num_ftrs_q, dim)
                self.encoder_k.classifier = nn.Linear(num_ftrs_k, dim)
            elif self.encoder_q.__class__.__name__.lower() == 'densenet':
                num_ftrs_q = self.encoder_q.classifier.in_features
                num_ftrs_k = self.encoder_k.classifier.in_features
                self.encoder_q.classifier = nn.Linear(num_ftrs_q, dim)
                self.encoder_k.classifier = nn.Linear(num_ftrs_k, dim)
            # AIHC hack - brute force replacement
        else:
            self.encoder_q = base_encoder(num_classes=dim)
            self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            if self.encoder_q.__class__.__name__.lower() == 'resnet':
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            elif self.encoder_q.__class__.__name__.lower() == 'mnasnet':
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                self.encoder_q.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.classifier)
                self.encoder_k.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.classifier)
            elif self.encoder_q.__class__.__name__.lower() == 'densenet':
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                self.encoder_q.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.classifier)
                self.encoder_k.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.classifier)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # TODO: set initial queue to random?
        self.register_buffer("queue_id", -1 * torch.ones(K))
        self.register_buffer("queue_study", -1 * torch.ones(K))
        self.register_buffer("queue_lat", -1 * torch.ones(K))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, meta_info):
        key_id, _ = meta_info['id']
        key_study, _ = meta_info['study']
        key_lat, _ = meta_info['lat']

        # gather keys before updating queue
        keys = concat_all_gather(keys)
        key_id = concat_all_gather(key_id)
        key_study = concat_all_gather(key_study)
        key_lat = concat_all_gather(key_lat)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys, key_id, key_study at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()
        self.queue_id[ptr:ptr + batch_size] = key_id
        self.queue_study[ptr:ptr + batch_size] = key_study
        self.queue_lat[ptr:ptr + batch_size] = key_lat

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, meta_info):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        weight = None
        mask = None
        index1 = None
        index2 = None
        frac = None
        # negative pairing strategy: laterality
        if self.hard_negative != 'random':
            # mask: NxK
            # mask easy negatives (set to True)
            mask_neg = self.select_hard_negative(meta_info)
            # mask positives in the negatives (set to True)
            mask_pos = self.mask_positive(meta_info)
            # exclude easy negatives and pos pairs
            mask = mask_neg | mask_pos
            if self.negative_laterality_only:
                logits[:, 1:][mask] = float('-inf')
            elif self.target_hard_negative:
                weight = self.reweight_logits(logits, mask_neg, meta_info)
            # subsample front/front pairs
            elif self.subsample:
                # mask unwanted queues (subsample) for front query (set to True)
                mask_subsample = self.sample_front(mask_neg, meta_info)
                mask = mask | mask_subsample
                logits[:, 1:][mask] = float('-inf')
            elif self.append_hard_negative:
                if (self.queue_lat == -1).sum() == 0:
                    _, query_lat = meta_info['lat']
                    sel_query_f = (1 - query_lat).bool()
                    sel_query_l = query_lat.bool()
                    num_appended = int(
                        self.append_hard_negative * self.K)
                    # generate fraction/index for synthetic samples
                    if self.synthesize_hard_samples:
                        frac, index1, index2 = sample_synthetic_index(
                            num_appended)
                        total_append_size = 2 * num_appended
                    else:
                        total_append_size = num_appended
                    # tensor for appending additonal logits
                    logits_append = logits.new_zeros(
                        logits.shape[0], total_append_size)
                    # frontal query
                    if sel_query_f.sum() > 0:
                        # select frontal logits
                        _, sel_queue_index_f = self.sample_negative_pairs(
                            mask_neg, meta_info, num_appended, 'front')
                        logits_front = logits[sel_query_f,
                                              1:][:, sel_queue_index_f]

                        if self.synthesize_hard_samples:
                            synthesis_logits_f = frac * logits_front[
                                :, index1] + (1 - frac) * logits_front[
                                :, index2]
                            queue_index1 = sel_queue_index_f[index1]
                            queue_index2 = sel_queue_index_f[index2]
                            synthesis_logits_f = self.normalize_synthesis(
                                queue_index1, queue_index2, frac,
                                synthesis_logits_f)
                            logits_front = torch.cat(
                                [logits_front, synthesis_logits_f], dim=1)

                        logits_append[sel_query_f, :] = logits_front

                    if sel_query_l.sum() > 0:
                        # select lateral logits
                        _, sel_queue_index_l = self.sample_negative_pairs(
                            mask_neg, meta_info, num_appended, 'lateral')

                        logits_lateral = logits[sel_query_l,
                                                1:][:, sel_queue_index_l]
                        # generate synthetic samples
                        if self.synthesize_hard_samples:
                            synthesis_logits_l = frac * logits_lateral[
                                :, index1] + (1 - frac) * logits_lateral[
                                :, index2]
                            queue_index1 = sel_queue_index_l[index1]
                            queue_index2 = sel_queue_index_l[index2]
                            synthesis_logits_l = self.normalize_synthesis(
                                queue_index1, queue_index2, frac,
                                synthesis_logits_l)
                            logits_lateral = torch.cat(
                                [logits_lateral, synthesis_logits_l], dim=1)

                        logits_append[sel_query_l, :] = logits_lateral

                    if sel_query_l.sum() > 0 or sel_query_f.sum() > 0:
                        # append the logits
                        logits = torch.cat([logits, logits_append], dim=1)
            else:
                raise Exception("invalid negative pairing method")

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, meta_info)

        return logits, labels, weight

    def normalize_synthesis(self, queue_index1, queue_index2, frac,
                            synth_logits):
        q1 = self.queue[:, queue_index1] * frac
        q2 = self.queue[:, queue_index2] * (1. - frac)
        assert q1.shape == q2.shape
        dot_prod = (q1 * q2).sum(0)
        norm_1 = (q1 * q1).sum(0)
        norm_2 = (q2 * q2).sum(0)
        norm_pairs = torch.sqrt(2 * dot_prod + norm_1 + norm_2)
        normalized_synth_logits = synth_logits / norm_pairs

        return normalized_synth_logits

    @ torch.no_grad()
    def select_hard_negative(self, meta_info):
        """return a mask that select the easy negatives pairs (set to 1) that
        are to be excluded.
        Args:
            meta_info(dict)
        returns:
            mask(torch.tensor(bool)): True if elements needs to be masked out
            and False otherwise
        """
        mask = None
        # mask out different laterality (i.e. easy negative)
        if self.hard_negative == "lateral":
            _, query_lat = meta_info['lat']
            mask = query_lat.unsqueeze(1) != self.queue_lat.unsqueeze(0)
        else:
            NotImplementedError("no prescribed hard negative pairing")

        # Debug
        if False:
            # print("percentage of chosen negative pairs " +
            #       f"{1 - mask.sum(1) / float(self.K)}")
            print(
                f"perc number of lat in queue {(self.queue_lat == 1).sum() / float(self.K)} ")
            print(
                f"perc number of front in queue {(self.queue_lat == 0).sum() / float(self.K)} ")
            # print(f"query lat {query_lat}")
            # print("\n")

        return mask

    @ torch.no_grad()
    def mask_positive(self, meta_info):
        """return a mask for excluding positive pairs within negative-pair
        selection
        Args:
            meta_info ([type]):

        Returns:
            mask:
        """
        mask = None
        same_study = self.positive_strat['same_study']
        diff_study = self.positive_strat['diff_study']
        same_lat = self.positive_strat['same_lat']
        diff_lat = self.positive_strat['diff_lat']

        _, query_id = meta_info['id']
        # mask same id
        mask_id = query_id.unsqueeze(1) == self.queue_id.unsqueeze(0)

        # same study and all laterality positive
        if same_study and not same_lat and not diff_lat:
            _, query_study = meta_info['study']
            # mask same study
            mask_study = query_study.unsqueeze(
                1) == self.queue_study.unsqueeze(0)
            # mask same id and same study
            mask = mask_id & mask_study
        else:
            NotImplementedError('Only use same study and no laterality' +
                                'positive strategy')

        return mask

    @ torch.no_grad()
    def reweight_logits(self, logits, mask_neg, meta_info):
        """compute weight matrix for the logits
        meta_info['lat']:
            front: 0
            lateral: 1

        Args:
            logits ([type]): [description]
            meta_info ([type]): [description]

        Returns:
            weight: [description]
        """
        query_info = None

        if (self.queue_lat == -1).sum() > 0:
            return torch.ones_like(logits, dtype=torch.float)

        if self.hard_negative == "lateral":
            _, query_info = meta_info['lat']
        else:
            NotImplementedError("no prescribed hard negative pairing")

        # lateral: w_f, w_l = front, lateral
        # target weight as coef of exp(logits)
        # weight for hard negative
        target_easy_negative = 1 - torch.tensor(self.target_hard_negative)
        perc_lat = self.queue_lat.sum() / self.K
        perc_front = 1. - perc_lat
        assert perc_lat > 0 and perc_front > 0
        # scaling for frontal query
        hard_negative_query_f = perc_front
        easy_negative_query_f = perc_lat
        w_f_h, w_f_e = get_weight(
            target_easy_negative, easy_negative_query_f,
            hard_negative_query_f)
        # scaling for lateral query
        hard_negative_query_l = perc_lat
        easy_negative_query_l = perc_front
        w_l_h, w_l_e = get_weight(
            target_easy_negative, easy_negative_query_l,
            hard_negative_query_l)
        # apply weight to logits based on the laterality pairing
        mask_neg = mask_neg.float()
        weight_ff = (1 - query_info).unsqueeze(1) * (1. - mask_neg) * w_f_h
        weight_fl = (1 - query_info).unsqueeze(1) * mask_neg * w_f_e
        weight_ll = query_info.unsqueeze(1) * (1. - mask_neg) * w_l_h
        weight_lf = query_info.unsqueeze(1) * mask_neg * w_l_e
        weight = weight_fl + weight_ff + weight_lf + weight_ll

        # set weight for positive pair to 1
        weight_total = torch.ones_like(logits)
        weight_total[:, 1:] = weight

        return weight_total

    @ torch.no_grad()
    def sample_front(self, mask_neg, meta_info):
        mask = None
        assert self.K == 24576, "only test for K= 24576"
        num_include = 3600

        if (self.queue_lat == -1).sum() > 0:
            return torch.zeros_like(mask_neg, dtype=torch.bool)

        if self.hard_negative == "lateral":
            _, query_info = meta_info['lat']
            f = torch.argmin(query_info)
            queue_f = (1. - mask_neg[f, :].int()).bool()
            index_f = torch.arange(self.K)[queue_f]
            index_sel = index_f[torch.multinomial(
                torch.ones_like(index_f, dtype=float), num_include)]
            mask = torch.ones_like(mask_neg, dtype=bool)
            mask[:, index_sel] = False
            mask[query_info.bool(), :] = False
        else:
            NotImplementedError("no subsample for the hard_negative")

        return mask

    @ torch.no_grad()
    def sample_negative_pairs(self, mask_neg, meta_info, num_samples,
                              laterality):
        """Warning: the return mask defines elements to be EXCLUDED.  To
        include the selected elements, apply the return mask by
        (1 - mask.int()).bool()

        Args:
            mask_neg ([type]): [description]
            meta_info ([type]): [description]
            laterality ([type]): [description]

        Returns:
            mask: [description]
        """
        mask = None
        queue_index_sel = None
        # set mask to all False (i.e. include all elements) until queue is fully
        # populated.
        if (self.queue_lat == -1).sum() > 0:
            return torch.zeros_like(mask_neg, dtype=torch.bool)

        if self.hard_negative == "lateral":
            _, query_info = meta_info['lat']
            if laterality == 'front':
                # select a front index in the query
                query_index = torch.argmin(query_info)
                # used to set lateral query to True (i.e. excluded)
                query_sel = query_info.bool()
            else:
                # select a lateral index in the query
                query_index = torch.argmax(query_info)
                # used to set frontal query to True (i.e. excluded)
                query_sel = (1 - query_info).bool()

            queue_hard = (1. - mask_neg[query_index, :].int()).bool()
            queue_hard_index = torch.arange(self.K)[queue_hard]

            # insuffcient amount of hard pairs in the queue
            if num_samples > queue_hard.sum():
                raise Exception(
                    "insufficient number of hard examples in the queues" +
                    " should be resolved by increasing the queue size K"
                    " to at least 2048")
            # sample hard queue indices
            queue_index_sel = queue_hard_index[torch.multinomial(
                torch.ones_like(queue_hard_index, dtype=float), num_samples)]
            mask = torch.ones_like(mask_neg, dtype=bool)
            # select the hard queue samples to be included (i.e. set False)
            mask[:, queue_index_sel] = False
            # exclude the query samples (i.e. set True)
            mask[query_sel, :] = True
        else:
            NotImplementedError("no subsample for the hard_negative")

        return mask, queue_index_sel


# utils
@ torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@ torch.no_grad()
def get_weight(target_easy_negative, easy_negative, hard_negative):
    """
    Args:
        target_easy_negative ([type]): [description]
        easy_negative ([type]): [description]
        hard_negative ([type]): [description]

    Returns:
        w_h, w_e: scaling factor for hard and easy and negative for achieving the
        target_easy_negative
    """
    w_e = target_easy_negative / easy_negative
    transfer_weight = easy_negative - target_easy_negative
    if transfer_weight < 0:
        warnings.warn(
            "Transfering weight from hard negative to easy negative")
    w_h = 1 + transfer_weight / hard_negative
    return w_h, w_e


@torch.no_grad()
def sample_synthetic_index(num_appended):

    frac = torch.rand(num_appended).unsqueeze(0).cuda()
    index1 = list(range(num_appended))
    index2 = index1.copy()
    random.shuffle(index1)
    random.shuffle(index2)

    return frac, index1, index2
