import torch.nn as nn
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from functools import reduce
import torchvision.models as models
import cv2
from torch.autograd import Variable

from training.perceptual_losses.nets import resnet50, load_state_dict


class VGGPerceptualLoss(torch.nn.Module):
    """
    """

    def __init__(self, 
            perceptual=True,
            contextual=False,
            # perc_layers=["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
            # original perceptual loss implementation:
            perc_layers=["relu1_2", "relu2_4", "relu3_4", "relu4_4"],
            perc_weights=None,      
            ctx_layers=["relu3_2",  "relu4_2"],
            ctx_weights=None,      
            vgg_type='vgg16',
            resize=True, 
        ):

        """
        TODO loss_weights: if None, all '1.'
        Note that both perceptual and contextual can be set at the same time,
        this is to avoid computing feature maps twice
        """

        super(VGGPerceptualLoss, self).__init__()
        assert perceptual or contextual, "Set at least one"


        if vgg_type == 'vgg16':
            self.vgg = models.vgg16(pretrained=True).features.eval()#.cuda()
        elif vgg_type == 'vgg19':
            self.vgg = models.vgg19(pretrained=True).features.eval()#.cuda()
        else:
            raise NotImplementedError("Implement")

        self.resize = resize
        self.perceptual = perceptual
        self.contextual = contextual

        # setup contextual loss
        if contextual:
            self.ctx_bias = 1.0
            self.ctx_nn_stretch_sigma = 0.5
            # self.ctx_lambda_style = 1.0
            # self.ctx_lambda_content = 1.0

        # setup weights for each layer
        self.perc_weights = None
        self.ctx_weights = None

        if perc_weights is not None:
            perc_name2weight = { perc_layers[i] : perc_weights[i] for i in range(len(perc_layers))}
            # mapping layer index -> weight
            self.perc_weights = {}

        if ctx_weights is not None:
            ctx_name2weight = { ctx_layers[i] : ctx_weights[i] for i in range(len(ctx_layers))}
            self.ctx_weights = {}

        # transforms
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        # required layers
        if perceptual:
            self.loss_layers = set(perc_layers)
        else:
            self.loss_layers = set(ctx_layers)

        if perceptual and contextual:
            self.loss_layers = self.loss_layers.union( set(ctx_layers) )
            
        self.all_layer_names = []
        self.all_blocks = []
        self.perc_blocks = [] if perceptual else None
        self.ctx_blocks = [] if contextual else None

        self.target=None

        ci = 1
        ri = 0

        for i, layer in enumerate(self.vgg.children()):
            # name layer
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.all_layer_names.append(name)

            # save
            if name in self.loss_layers:
                # keeps indices to all blcoks (perceptual and contextual)
                self.all_blocks.append(i)

                if perceptual and name in perc_layers:
                    self.perc_blocks.append(i)
                    if self.perc_weights is not None:
                        self.perc_weights[i] = perc_name2weight[name]

                if contextual and name in ctx_layers:
                    self.ctx_blocks.append(i)
                    if self.ctx_weights is not None:
                        self.ctx_weights[i] = ctx_name2weight[name]

            # x = layer(x)
            # out[name] = x

        # check we could recover all specified layers
        error = False
        if perceptual and len(perc_layers) != len(self.perc_blocks):
            print ("")
            print ("Perceptual - Input layers:")
            print (perc_layers)
            print ("Recovered layers:")
            for k in self.perc_blocks:
                print (k, self.all_layer_names[k])
            error=True

        if contextual and len(ctx_layers) != len(self.ctx_blocks):
            print ("")
            print ("Contextual - Input layers:")
            print (ctx_layers)
            print ("Recovered layers:")
            for k in self.ctx_blocks:
                print (k, self.all_layer_names[k])
            error=True

        if error:
            print ("")
            print ("Layer nanes:")
            for n in self.all_layer_names:
                print("", n)
            raise Exception("Some layers were not found")


    # -------------
    def forward(self, input, target, average=True, keep_precomputed_target=False):
        """ don't use keep_precomputed_target!! needs retain_graph=True.... """

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        bsize = input.size(0)

        x = (input-self.mean) / self.std
        y = (target-self.mean) / self.std

        if self.resize:
            x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
            y = self.transform(y, mode='bilinear', size=(224, 224), align_corners=False)
        
        # x = input
        # y = target

        # x = x - self.mean
        # x = x/self.std

        # y = y - self.mean
        # y = y/self.std

        curr_block = 0
        loss = None

        if keep_precomputed_target and self.target is None:
            targets = {}

        for i, layer in enumerate(self.vgg.children()):

            x = layer(x)
            if not keep_precomputed_target or self.target is None:
                y = layer(y)

            if i == self.all_blocks[curr_block]:
                
                perc_loss = 0.
                ctx_loss = 0.

                if keep_precomputed_target:
                    if self.target is None:
                        targets[i] = y
                    else:
                        y = self.target[i]

                if self.perceptual and i in self.perc_blocks:
                    reduction = 'mean' if average else 'none'
                    perc_loss = torch.nn.functional.l1_loss(x, y, reduction=reduction)
                    
                    if self.perc_weights is not None:
                        w = self.perc_weights[i]
                        perc_loss = w*perc_loss 


                if self.contextual and i in self.ctx_blocks:
                    # original code was with sum, TODO CHECK
                    # reduction = 'sum' if average else 'none'
                    reduction = 'mean' if average else 'none'
                    ctx_loss = self.contextual_loss(x, y, reduction=reduction)
                    
                    if self.ctx_weights is not None:
                        w = self.ctx_weights[i]
                        ctx_loss = w*ctx_loss 

                curr_loss = perc_loss + ctx_loss
                
                if self.perceptual and self.contextual:
                    curr_loss /= 2.

                if not average:
                    # just don't average at the batch level
                     curr_loss = curr_loss.view(bsize, -1).mean(1)

                if loss is None:
                    loss = curr_loss
                else:
                    loss += curr_loss

                curr_block += 1

                if curr_block == len(self.all_blocks):
                    break

        if keep_precomputed_target and self.target is None:
            # for next time
            self.target = targets

        return loss


    # -------------
    def contextual_loss(self, gen, tar, reduction='mean'):
        """
        TODO cite, from ID-MRF code
        """
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self._patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)

        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self._compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self._exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        res = -torch.log(div_mrf)

        if reduction == 'mean':
            # size: bsize
            res = res.mean()
        elif reduction == 'sum':
            # size: bsize
            res = torch.sum(res)
        return res


    # -------------
    def _sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def _patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def _compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def _exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.ctx_bias - scaled_dist)/self.ctx_nn_stretch_sigma)
        self.cs_NCHW = self._sum_normalize(dist_before_norm)
        return self.cs_NCHW



# =============================================================================
class VGGFace2Loss(nn.Module):
    def __init__(self, pretrained_data='vggface2'):
        super(VGGFace2Loss, self).__init__()
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval().cuda()
        load_state_dict(self.reg_model, pretrained_data)
        self.mean_bgr = torch.tensor([91.4953, 103.8827, 131.0912]).cuda()

    #     self._freeze_layer(self.reg_model)
    # def _freeze_layer(self, layer):
    #     for param in layer.parameters():
    #         param.requires_grad = False
    def reg_features(self, x):
        # out = []
        margin=10
        x = x[:,:,margin:224-margin,margin:224-margin]
        # x = F.interpolate(x*2. - 1., [224,224], mode='nearest')
        x = F.interpolate(x*2. - 1., [224,224], mode='bilinear')
        # import ipdb; ipdb.set_trace()
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # import ipdb;ipdb.set_trace()
        img = img[:, [2,1,0], :, :].permute(0,2,3,1) * 255 - self.mean_bgr
        img = img.permute(0,3,1,2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True):
        gen = self.transform(gen)
        tar = self.transform(tar)

        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)
        # loss = ((gen_out - tar_out)**2).mean()
        loss = self._cos_metric(gen_out, tar_out).mean()
        return loss


if __name__ == '__main__':
    im1 = torch.rand(2,3,256,256).to('cuda')
    im2 = torch.rand(2,3,256,256).to('cuda')

    id_loss = VGGFace2Loss(
        pretrained_data='/is/cluster/scratch/pghosh/GIF_resources/input_files/'
                        'perceptual_loss_resources/resnet50_ft_weight.pkl').to('cuda')
    perceptual_loss = VGGPerceptualLoss(
                                resize=(im1.shape[2] != 224), 
                                vgg_type='vgg16',
                                perceptual=True,
                                contextual=False).to('cuda')

    #https://arxiv.org/abs/1803.02077
    contextual_loss = VGGPerceptualLoss(
                                resize=(im1.shape[2] != 224), 
                                vgg_type='vgg16',
                                perceptual=False,
                                contextual=True).to('cuda')

    loss = id_loss(im1, im2)
    print ("ID loss:", loss.item())

    loss = perceptual_loss(im1, im2)
    print ("Perceptual loss: ", loss.item())

    loss = contextual_loss(im1, im2)
    print ("Contextual loss: ", loss.item())
