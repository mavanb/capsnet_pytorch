import torch
import torch.nn as nn
from utils import squash, init_weights, flex_profile, get_device, calc_entropy, batched_index_select


class DynamicRouting(nn.Module):

    def __init__(self, j, i, n, softmax_dim, bias_routing, sparse_threshold, sparsify, sparse_topk):
        super().__init__()
        self.soft_max = torch.nn.Softmax(dim=softmax_dim)
        self.j = j
        # self.i = i
        self.n = n

        # init depends on batch_size which depends on input size, declare dynamically in forward. see:
        # https://discuss.pytorch.org/t/dynamic-parameter-declaration-in-forward-function/427/2
        self.b_vec = None

        if bias_routing:
            b_routing = nn.Parameter(torch.zeros(j, n))
            b_routing.data.fill_(0.1)
            self.bias = b_routing
        else:
            self.bias = None

        # that can be implemented to enable analysis at end of each routing iter
        self.log_function = None
        self.sparse_threshold = sparse_threshold
        self.sparse_topk = [float(i) for i in sparse_topk.split(";")]
        self.sparsify = sparsify

    @flex_profile
    def forward(self, u_hat, iters):

        # check if enough topk values ratios are given in the configuration
        assert len(self.sparse_topk) + 1 >= iters, "Please specify for each update routing iter the sparse top k. " \
                            "Example: routing iters: 3, sparse_topk = 0.4;0.4"

        b = u_hat.shape[0]
        self.i = u_hat.shape[2]
        routing_stats = {}

        # todo: previously b_vec as only init once, but than the class must always have the same input shape
        # todo: it seems that init does not cost much time, so temporily do it this way
        # if self.b_vec is None:
        #     self.b_vec = torch.zeros(b, self.j, self.i, device=get_device(),  requires_grad=False)
        self.b_vec = torch.zeros(b, self.j, self.i, device=get_device(), requires_grad=False)
        b_vec = self.b_vec

        # track entropy of b_vec per iter
        routing_stats["H_c_vec"] = {}

        for index in range(iters):

            # softmax over j, weight of all predictions should sum to 1
            c_vec = self.soft_max(b_vec)

            # compute entropy of weight distribution of all capsules
            # routing_stats["H_c_vec"][index] = calc_entropy(c_vec, dim=1).mean().item()
            routing_stats["H_c_vec"][index] = 0

            # created unsquashed prediction for parents capsules by a weighted sum over the child predictions
            # in einsum: bij, bjin-> bjn
            # in matmul: bj1i, bjin = bj (1i)(in) -> bjn
            s_vec = torch.matmul(c_vec.view(b, self.j, 1, self.i), u_hat).squeeze()

            # add bias to s_vec
            if type(self.bias) == torch.Tensor:
                s_vec_bias = s_vec + self.bias

                # don't add a bias to capsules that have no activation add all
                # check which capsules where zero
                reset_mask = (s_vec.sum(dim=2) == 0)
                # set them back to zero again
                s_vec_bias[reset_mask, :] = 0
            else:
                s_vec_bias = s_vec
            v_vec = squash(s_vec_bias)

            if index < (iters - 1):  # skip update last iter
                # in einsum: "bjin, bjn-> bij", inner product over n
                # in matmul: bji1n, bj1n1 = bji (1n)(n1) = bji1
                # note: use x=x+1 instead of x+=1 to ensure new object creation and avoid inplace operation
                b_vec = b_vec + torch.matmul(u_hat.view(b, self.j, self.i, 1, self.n),
                                             v_vec.view(b, self.j, 1, self.n, 1)).view(b, self.j, self.i)

                if self.sparsify == "nodes_threshold":
                    b_vec, routing_stats = self.sparsify_nodes_threshold(b_vec, index, iters, routing_stats)
                elif self.sparsify == "nodes_topk":
                    b_vec, routing_stats = self.sparsify_nodes_topk(b_vec, index, iters, routing_stats)
                elif self.sparsify == "edges_threshold":
                    b_vec, routing_stats = self.sparsify_edges_threshold(b_vec, index, iters)
                elif self.sparsify == "edges_topk":
                    b_vec, routing_stats = self.sparsify_edges_topk(b_vec, index, iters, routing_stats)
                elif self.sparsify == "edges_random":
                    b_vec, routing_stats = self.sparsify_edges_random(b_vec, index, iters, routing_stats)

            if self.log_function:
                self.log_function(index, u_hat, b_vec, c_vec, v_vec, s_vec, s_vec_bias)
        return v_vec, routing_stats

    def sparsify_nodes_topk(self, b_vec, index, iters, routing_stats):

        current_mask_rato = self.sparse_topk[index]

        # number of elements to mask: mask rato times abs number
        mask_count = int(current_mask_rato * self.j)

        if mask_count > 0:

            # set prev -inf to inf to ignore them (only after first iter)
            if index > 0:
                prev_inf_mask = b_vec == float("-inf")
                b_vec[prev_inf_mask] = float("inf")

            # incoming weight of parent capsule (average over childs)
            z_j = b_vec.sum(dim=2) / self.i

            # take bottomk smallest values
            val, _ = torch.topk(z_j, mask_count, largest=False, sorted=False, dim=1)

            # get largest smallest value
            kthvalues, _ = torch.max(val, dim=1, keepdim=True)

            # mask all equal or smaller than largest smallest value
            delete_values = torch.le(z_j, kthvalues)

            # set prev -inf back (only after first iter)
            if index > 0:
                b_vec[prev_inf_mask] = float("-inf")

            b_vec[delete_values, :] = float("-inf")

            assert ((b_vec == float("-inf")).sum(dim=1) == self.j).nonzero().shape == torch.Size([0]), "Too many topk " \
                        "nodes are sparsified, all j cols are now -inf. Set sparse_topk lower."

        return b_vec, routing_stats

    def sparsify_edges_random(self, b_vec, index, iters, routing_stats):
        #todo| make sure that each column (child) has the same amount of masked elements. Requires shuffling, randperm
        #todo| works only in one dimension

        current_mask_rato = self.sparse_topk[index]

        if current_mask_rato > 0:

            random_ratios = torch.rand_like(b_vec, device=get_device(), requires_grad=False)

            mask = torch.le(random_ratios, current_mask_rato)

            ## Random edges mask with deterministic mask ratio, but same for each batch,
            ## a problem could be that gradient signal in each batch is to specialized
            # b = b_vec.shape[0]
            # total = self.j * self.i
            # ones_count = round(current_mask_rato * total)
            # zero_count = total - ones_count
            #
            # x = torch.zeros(zero_count, dtype=torch.uint8, device=get_device())
            # y = torch.ones(ones_count, dtype=torch.uint8, device=get_device())
            # xy = torch.cat((x, y))
            # mask = xy[torch.randperm(total)].view(1, self.j, self.i)

            valid_mask = self.full_inf(b_vec, mask, method="no_mask")
            b_vec[valid_mask] = float("-inf")

        return b_vec, routing_stats

    def sparsify_edges_topk(self, b_vec, index, iters, routing_stats, resolve_full_inf="no_mask"):

        current_mask_rato = self.sparse_topk[index]

        # number of elements to mask: mask rato times abs number
        mask_count = int(current_mask_rato * self.j)

        # mask_count of 0 means no sparsify, thus skip
        if mask_count > 0:

            # set all non top k to -inf, not trivial. topk returns an indices, which can't be used to index on b_vec (thus,
            # b_vec[indices] does't work). Therefore, we take the kth value and use torch.ge to get the binary tensor, which
            # can be used to index on. However, torch.kthvalue is does not work on cuda (https://github.com/pytorch/pytorch/
            # issues/2134). Instead, we take the max of topk (is largest excluded value).

            # set prev -inf to inf to ignore them
            prev_inf_mask = b_vec == float("-inf")
            b_vec[prev_inf_mask] = float("inf")

            # take bottomk smallest values
            val, _ = torch.topk(b_vec, mask_count, largest=False, sorted=False, dim=1)

            # get largest smallest value
            kthvalues, _ = torch.max(val, dim=1, keepdim=True)

            # mask all equal or smaller than largest smallest value
            mask = torch.le(b_vec, kthvalues)

            # set prev -inf back
            b_vec[prev_inf_mask] = float("-inf")

            # check full inf cols
            self.full_inf(b_vec, mask, method="raise")


            # finally, use the valid mask. note: doing this valid mask check on c_vec gives an inplace error
            b_vec[mask] = float("-inf")

        return b_vec, routing_stats

    def sparsify_edges_topk_old(self, b_vec, index, iters, routing_stats, resolve_full_inf="no_mask"):
        """ In this sparsify edges topk implementation, the smallest edges of all edges in a layer are dropped.
        This means that for childs some more edges are than for others, potentially, all edges may be dropped.
        In the new implemention sparsify_edges_topk, we only drop columnwise (same amount per child capsule).
        """

        # reshape to allow top k over i and j dim
        b_vec_flat = b_vec.view(-1, self.i * self.j)

        current_mask_rato = self.sparse_topk[index]

        # number of elements to mask: mask rato times abs number
        mask_count = int(current_mask_rato * self.j * self.i)

        # mask_count of 0 means no sparsify, thus skip
        if mask_count > 0:

            # set all non top k to -inf, not trivial. topk returns an indices, which can't be used to index on b_vec (thus,
            # b_vec[indices] does't work). Therefore, we take the kth value and use torch.ge to get the binary tensor, which
            # can be used to index on. However, torch.kthvalue is does not work on cuda (https://github.com/pytorch/pytorch/
            # issues/2134). Instead, we take the max of topk (is largest excluded value).

            # take bottomk smallest values
            val, _ = torch.topk(b_vec_flat, mask_count, largest=False, sorted=False, dim=1)

            # get largest smallest value
            kthvalues, _ = torch.max(val, dim=1, keepdim=True)

            # mask all equal or smaller than largest smallest value
            flat_mask = torch.le(b_vec_flat, kthvalues)

            # reshape back to original b_vec shape
            mask = flat_mask.view(-1, self.j, self.i)

            # resolve full inf cols
            valid_mask = self.full_inf(b_vec, mask, method=resolve_full_inf)

            # finally, use the valid mask. note: doing this valid mask check on c_vec gives an inplace error
            b_vec[valid_mask] = float("-inf")

        return b_vec, routing_stats

    def full_inf(self, b_vec, mask, method):

        # compute the entries that do not have only -inf on the j colums
        # the entries with inf in b_vec if the mask is applied, | gives OR operator
        inf_entries = mask | (b_vec == float("-inf"))

        # entries where the full j col is -inf
        # indices_non_full_inf_cols = (inf_entries.sum(dim=1) != self.j).view(b, 1, self.i)
        b = b_vec.shape[0]
        full_inf = (inf_entries.sum(dim=1) == self.j).view(b, 1, self.i)

        # resolve full inf columns by mask all but the largest value
        if method == "only_max":

            # entries that have the max value in b_vec in column j
            # max returns indices, to get bytetensor format use the values instead with eq comparision
            maxvalues_b_vec = torch.eq(torch.max(b_vec, dim=1)[0].view(b, 1, self.i), b_vec)

            # entries that should be used to correct the current mask, if 1 they not allowed to set to -inf
            correcting_mask = full_inf & maxvalues_b_vec

            # mask, but keep the largest value in column j if the whole column is -inf otherwise
            valid_mask = mask & (correcting_mask ^ 1)

        # resolve full inf columns by not masking this column at all
        elif method == "no_mask":
            # other mask method, don't apply mask on this col at all if causes full inf column
            valid_mask = mask & (full_inf ^ 1)
        elif method == "raise":
            if full_inf.sum() > 0:
                raise ValueError("Sparsifying resulted in full inf column.")
        else:
            raise ValueError("Method to resolve full infinite j column does not exits.")

        return valid_mask

    def sparsify_edges_threshold(self, b_vec, index, iters):
        raise NotImplementedError

    def sparsify_nodes_threshold(self, b_vec, index, iters, routing_stats):
        """ Sparsify the nodes of the parse tree using a threshold on the incoming weights of the nodes.
        The threshold is compared against the incoming weights a node will have after taking the the
        softmax. Sparsify only at the last iteration. The resulting zero rows in c_vec make rows in s_vec zero, which
        result in small v_vec values. These are set to zero in the callback.

        Args:
            b_vec: (tensor) non-sparse current b_vec
            index: (int) current routing iter
            iters: (int) total routing iters

        Returns: (tuple): tuple containing:
                    arg1: (tensor) b_vec sparsified
                    arg2: (dict) routing stats
                    arg3: (tensor->tensor) callable to set v_vec completely to zero
        """
        # only sparsify on last update #todo: allow also on other iterations, avoid full j cols to be -inf though
        if index == (iters - 2):

            # incoming weight of parent capsule (average over childs)
            z_j = b_vec.sum(dim=2) / self.i

            # we set all rows to zero of which the average value gives a probability of less than the threshold after
            # taking the softmax: e^z_j / (sum_k e^z_k) < threshold   (z_i = incoming weight of parent)
            # gives: z_j < log(threshold) + log (sum_k e^z_k))

            # compute left hand side with log sum exp trick
            a, _ = torch.max(z_j, dim=1)
            exponent = z_j - a.view(-1, 1)
            lhs = torch.tensor(self.sparse_threshold, device=get_device()).log() + a + torch.log(
                torch.exp(exponent).sum(dim=1))

            # delete all incoming weight of parent j if smaller than lhs
            delete_values = (z_j < lhs.view(-1, 1))
            b_vec[delete_values, :] = float("-inf")

            # Now, compute some routing statistics

            # compute weights in c_vec space to allow comparision with the threshold
            cz_j = nn.functional.softmax(z_j, dim=1)

            # average incoming weight (average over parent)
            cz_avg = (cz_j.sum(dim=1) / self.j).view(-1, 1)

            # the deviation from the average per parent
            deviation_cz_j = (cz_j - cz_avg)

            # negative deviations
            neg_deviations = deviation_cz_j * (deviation_cz_j < 0).float()
            avg_neg_deviations = neg_deviations.mean()
            max_neg_deviations = neg_deviations.min(dim=1)[0].mean(dim=0)

            # average number of nodes masked
            b = b_vec.shape[0]
            mask_rato = len(b_vec[b_vec == float("-inf")]) / (self.j * b * self.i)

            routing_stats["mask_rato"] = mask_rato
            routing_stats["avg_neg_devs"] = avg_neg_deviations.item()
            routing_stats["max_neg_devs"] = max_neg_deviations.item()

        return b_vec, routing_stats


class LinearPrimaryLayer(nn.Module):

    def __init__(self, in_features, out_capsules, vec_len):
        super().__init__()
        self.out_capsules = out_capsules
        self.vec_len = vec_len
        self.linear = nn.Linear(in_features, out_capsules * vec_len, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return squash(x.view(-1, self.out_capsules, self.vec_len))


class Conv2dPrimaryLayer(nn.Module):

    def __init__(self, in_channels, out_channels, vec_len, squash_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vector_length = vec_len

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * vec_len, kernel_size=9, stride=2,
                              bias=True)
        self.conv = init_weights(conv)

        self.squash_dim = squash_dim

    def forward(self, input):
        """
        :param input: [b, c, h, w]
        :return: [b, c, h, w, vec]
        """
        features = self.conv(input)     # [b, out_c*vec_len, h, w)
        _, _, h, w = features.shape
        caps_raw = features.contiguous().view(-1, self.out_channels, self.vector_length, h, w)      # [b, c, vec, h, w]
        caps_raw = caps_raw.permute(0, 1, 3, 4, 2)  # [b, c, h, w, vec]

        return squash(caps_raw, dim=self.squash_dim)


class DenseCapsuleLayer(nn.Module):

    def __init__(self, in_capsules, out_capsules, vec_len_in, vec_len_out, stdev):
        super(DenseCapsuleLayer, self).__init__()

        self.i = in_capsules
        self.j = out_capsules
        self.m = vec_len_in
        self.n = vec_len_out

        self.W = nn.Parameter(stdev * torch.randn(1, out_capsules, in_capsules, vec_len_out, vec_len_in))

    def forward(self, input):
        b,i,m = input.shape
        n = self.n
        j = self.j
        assert i == self.i, "Unexpected number of childs as input"
        assert m == self.m, "Unexpected vector lenght as input"

        # the number of zero rows (m index all zero) in the input, for speed check only for the first item in batch
        zero_count = (input[0].sum(dim=1) == 0.0).sum().item()

        if zero_count > 0:

            non_zero_count = i - zero_count

            # chech which rows are not zero, and put in format to use for batch_index_select
            select_idx = (input.sum(dim=2) != 0.0).nonzero()[:, 1].view(b, non_zero_count)

            # expand W such that we can multiply batch elements with its own truncated W
            W = self.W.expand(b, self.j, self.i, self.n, self.m)
            W = batched_index_select(W, 2, select_idx)
            input = batched_index_select(input, 1, select_idx)
        else:
            W = self.W

        input = input.view(b, 1, input.shape[1], self.m, 1)

        # W: bjinm or 1jinm
        # input: b1jm1
        # matmul: bji(nm) * b1j(m1) = bjin1
        u_hat = torch.matmul(W, input).view(b, j, i - zero_count, n)

        return u_hat


class Conv2dCapsuleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Not implemented yet.")





