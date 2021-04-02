import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
import functools

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        weight = torch.FloatTensor([0.1])
        self.weight = torch.nn.Parameter(weight,requires_grad=True)
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=use_attention)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, inputs):
        ori_code_map = inputs[:, 4:, :, :]
        code_map_input = ori_code_map* torch.clamp(self.weight,min=0.001)
        mew_inputs = torch.cat([inputs[:, :4, :, :],code_map_input],1)
        return self.model(mew_inputs)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.use_attention = use_attention
        if use_attention:
            attention_conv = nn.Conv2d(outer_nc+input_nc, outer_nc+input_nc, kernel_size=1)
            attention_sigmoid = nn.Sigmoid()
            self.attention = nn.Sequential(*[attention_conv, attention_sigmoid])

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            ret = torch.cat([x, self.model(x)], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class StyleEncoder(nn.Module):
    def __init__(self, style_dim, norm_layer=nn.BatchNorm2d):
        super(StyleEncoder, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        ndf=64
        n_layers=6
        kw = 3
        padw = 0
        self.conv1f = PartialConv2d(3, ndf, kernel_size=kw, stride=2, padding=padw)
        self.relu1 = nn.ReLU(True)
        nf_mult = 1
        nf_mult_prev = 1

        n = 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv2f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm2f = norm_layer(ndf * nf_mult)
        self.relu2 = nn.ReLU(True)

        n = 2
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv3f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm3f = norm_layer(ndf * nf_mult)
        self.relu3 = nn.ReLU(True)

        n = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv4f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm4f = norm_layer(ndf * nf_mult)
        self.relu4 = nn.ReLU(True)

        n = 4
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv5f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.Conv2d(ndf * nf_mult, style_dim, kernel_size=1, stride=1)

    def forward(self, input, mask):
        """Standard forward."""
        xb = input
        mb = mask

        xb, mb = self.conv1f(xb, mb)
        xb = self.relu1(xb)
        xb, mb = self.conv2f(xb, mb)
        xb = self.norm2f(xb)
        xb = self.relu2(xb)
        xb, mb = self.conv3f(xb, mb)
        xb = self.norm3f(xb)
        xb = self.relu3(xb)
        xb, mb = self.conv4f(xb, mb)
        xb = self.norm4f(xb)
        xb = self.relu4(xb)
        xb, mb = self.conv5f(xb, mb)
        xb = self.avg_pooling(xb)
        s = self.convs(xb)
        return s


class BargainNetModel(BaseModel):


    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_tri', type=float, default=0.01, help='weight for triplet losses')
            parser.add_argument('--lambda_f2b', type=float, default=1.0, help='to ablate the triplet loss of /hat z_f and z_b')
            parser.add_argument('--lambda_ff2', type=float, default=1.0, help='to ablate the triplet loss of z_f and /hat z_f')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1','tri']
        self.visual_names = ['comp', 'real', 'harm', 'real_f', 'fake_f', 'mask', 'bg']
        self.model_names = ['E','G']
        
        self.style_dim = opt.style_dim
        netE = StyleEncoder(self.style_dim, norm_layer=nn.BatchNorm2d)
        self.netE = networks.init_net(netE, opt.init_type, opt.init_gain, self.gpu_ids)
        netG = UnetGenerator(opt.input_nc, opt.output_nc,8, opt.ngf, nn.BatchNorm2d, not opt.no_dropout,use_attention=True)
        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)
        self.relu = nn.ReLU()
        if self.isTrain:
            self.margin = opt.margin
            self.tripletLoss = nn.TripletMarginLoss(margin=self.margin, p=2)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr*opt.e_lr_ratio,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_E)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*opt.g_lr_ratio,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.iter_cnt = 0

    def set_input(self, input):
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.inputs = torch.cat([self.comp,self.mask],1).to(self.device)
        self.bg = 1.0 - self.mask
        self.real_f = self.real * self.mask

    def forward(self):
        self.bg_sty_vector = self.netE(self.real, self.bg)
        self.real_fg_sty_vector = self.netE(self.real, self.mask)
        self.bg_sty_map = self.bg_sty_vector.expand([1,self.style_dim,256,256])
        self.inputs_c2r = torch.cat([self.inputs,self.bg_sty_map],1)
        self.harm = self.netG(self.inputs_c2r)
        self.harm_fg_sty_vector = self.netE(self.harm, self.mask)

        self.comp_fg_sty_vector = self.netE(self.comp, self.mask)

        self.fake_f = self.harm * self.mask


    def backward(self):

        self.loss_L1 = self.criterionL1(self.harm, self.real)
        self.loss_tri = (self.tripletLoss(self.real_fg_sty_vector, self.harm_fg_sty_vector, self.comp_fg_sty_vector)*self.opt.lambda_ff2 +\
            self.tripletLoss(self.harm_fg_sty_vector, self.bg_sty_vector, self.comp_fg_sty_vector)*self.opt.lambda_f2b)* self.opt.lambda_tri 
        self.loss = self.loss_L1 + self.loss_tri 

        self.loss.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_E.step()
        self.optimizer_G.step()
