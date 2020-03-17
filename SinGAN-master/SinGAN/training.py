import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize
import numpy as np
import torch
import cv2
from pytorch_msssim import ssim, ms_ssim, MS_SSIM


def train(opt, Gs, Zs, reals, NoiseAmp):
    real_ = functions.read_image(opt)
    in_s = 0
    scale_num = 0
    real = imresize(real_, opt.scale1, opt)

    # 不同规格数据形成的列表
    reals = functions.creat_reals_pyramid(real, reals, opt)
    # print('reals', reals)  # 各个scale的图形形成的列表

    # plt.imsave('Output/real_scale_0.png', functions.convert_image_np(reals[0]), vmin=0, vmax=1)
    # plt.imsave('Output/real_scale_1.png', functions.convert_image_np(reals[1]), vmin=0, vmax=1)
    # plt.imsave('Output/real_scale_2.png', functions.convert_image_np(reals[2]), vmin=0, vmax=1)
    # plt.imsave('Output/real_scale_3.png', functions.convert_image_np(reals[3]), vmin=0, vmax=1)
    # plt.imsave('Output/real_scale_4.png', functions.convert_image_np(reals[4]), vmin=0, vmax=1)
    # plt.imsave('Output/real_scale_5.png', functions.convert_image_np(reals[5]), vmin=0, vmax=1)
    # plt.imsave('Output/real_scale_6.png', functions.convert_image_np(reals[6]), vmin=0, vmax=1)
    # plt.imsave('Output/real_scale_7.png', functions.convert_image_np(reals[7]), vmin=0, vmax=1)
    # plt.imsave('Output/real_scale_8.png', functions.convert_image_np(reals[8]), vmin=0, vmax=1)
    # plt.imsave('Output/real_scale_9.png', functions.convert_image_np(reals[9]), vmin=0, vmax=1)

    nfc_prev = 0

    # opt.stop_scale = 9   循环9次
    while scale_num < opt.stop_scale + 1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        print('opt.nfc', opt.nfc)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        print('opt.min_nfc', opt.min_nfc)
        if opt.fast_training:
            if (scale_num > 0) & (scale_num % 4 == 0):
                opt.niter = opt.niter // 2

        # out_是生成根路径
        opt.out_ = functions.generate_dir2save(opt)
        # outf是每个scale路径
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        # 保存原始图像
        plt.imsave('%s/original.png' % (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        # 在每个scale中保存real_scale
        plt.imsave('%s/real_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        # return netD, netG  目前的D和G，D_curr
        D_curr, G_curr = init_models(opt)
        if (nfc_prev == opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))

        # train_single_scale()返回：z_opt, in_s, netG
        z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)

        G_curr = functions.reset_grads(G_curr, False)
        print('G_curr', G_curr)
        G_curr.eval()
        print(G_curr.eval())
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr
    return


# 训练单次scale
def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt, centers=None):
    # print("Gs:", Gs)
    # Gs:scale尺度
    print("len(Gs):", len(Gs))

    # 获取当前scale的真实值
    real = reals[len(Gs)]

    opt.nzx = real.shape[2]  # +(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]  # +(opt.ker_size-1)*(opt.num_layer)
    # 接受野
    opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
    print(opt.receptive_field) # out: 3+2*4*1 = 11
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    print(pad_noise)  # pad_noise: 5
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    # ZeroPad2d 在输入的数据周围做zero-padding
    m_noise = nn.ZeroPad2d(int(pad_noise))
    print('m_noise', m_noise)  # ZeroPad2d(padding=(5, 5, 5, 5), value=0.0)
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha
    print(alpha)

    fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy])
    # print('fixed_noise', fixed_noise.shape)  #  torch.Size([1, 3, 76, 76])
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # 设置优化器
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

    # 绘画损失列表
    list_ssim = []
    list_ssim_1 = []
    list_ssim_2 = []
    list_ssim_3 = []
    list_ssim_4 = []

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    # 循环迭代
    for epoch in range(opt.niter):
        schedulerD.step()
        schedulerG.step()
        # 与运算
        if (Gs == []) & (opt.mode != 'SR_train'):
            # opt.nzx和opt.nzy是当前scale的尺寸
            z_opt = functions.generate_noise([1, opt.nzx, opt.nzy])
            # 扩充维度为(1, 3, opt.nzx, opt.nzy)
            z_opt = m_noise(z_opt.expand(1, 3, opt.nzx, opt.nzy))
            noise_ = functions.generate_noise([1, opt.nzx, opt.nzy])
            noise_ = m_noise(noise_.expand(1, 3, opt.nzx, opt.nzy))
        else:
            noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy])
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        # 更新判别器
        ###########################

        # Dsteps = 3
        for j in range(opt.Dsteps):
            # train with real 用真实图像训练
            netD.zero_grad()

            output = netD(real).to(opt.device)
            # print(netD)
            # print('output', output)  # 4维数据

            errD_real = -output.mean()  # -a
            # print('errD_real', errD_real) # -2.
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake 用虚假图像训练
            # 仅第一次训练用到z_prev(噪声)
            if (j == 0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s = prev
                    prev = m_image(prev)

                    # nc_z  3通道
                    z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    # print('z_prev', z_prev)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    # MSE 军方误差损失函数
                    criterion = nn.MSELoss()

                    # 均方根误差, 标准误差
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                    prev = m_image(prev)
                    z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
                    criterion = nn.MSELoss()
                    # 标准误差
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                prev = m_image(prev)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp * noise_ + prev

            # .detach()用于切断反向传播
            fake = netG(noise.detach(), prev)

            # 添加的ssim_loss
            ssim_loss = ssim(real, fake, data_range=255, size_average=True)

            output = netD(fake.detach())
            # 判别以后损失反向传播
            errD_fake = output.mean()
            # print('errD_fake', errD_fake)  # -0.0072
            errD_fake.backward(retain_graph=True)
            # 判别器_生成器_噪声
            D_G_z = output.mean().item()
            # print('D_G_z', D_G_z)

            # 梯度惩罚---------------------------------------------------------------------------
            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad)
            # print('gradient_penalty', gradient_penalty)
            # 梯度惩罚更新
            gradient_penalty.backward()

            # 损失函数：对抗损失 + 重构损失
            D_ssim_3 = errD_real + errD_fake + gradient_penalty
            errD = errD_real + errD_fake + gradient_penalty

            # print('item', errD.item())

            # D_ssim = 0.8 * (errD_real + errD_fake) + 0.68 * gradient_penalty + (1 - ssim_loss)
            # D_ssim_1 = 0.7 * (errD_real + errD_fake) + 0.6 * gradient_penalty + 1.2 * (1 - ssim_loss)
            # D_ssim_2 = 0.75 * (errD_real + errD_fake) + 0.6 * gradient_penalty + 1.2 * (1 - ssim_loss)
            # errD = (errD_real + errD_fake) + 0.5 * gradient_penalty + 1.4 * (1 - ssim_loss)
            # D_ssim_4 = 0.6 * (errD_real + errD_fake) + 0.5 * gradient_penalty + 1.4 * (1 - ssim_loss)

            # int_ssim = D_ssim.item()
            # int_ssim = round(int_ssim, 4)

            # int_ssim_1 = D_ssim_1.item()
            # int_ssim_1 = round(int_ssim_1, 4)
            #
            # int_ssim_2 = D_ssim_2.item()
            # int_ssim_2 = round(int_ssim_2, 4)


            int_ssim_3 = D_ssim_3.item()
            int_ssim_3 = round(int_ssim_3, 4)


            optimizerD.step()

        errDint = []
        errD2plot.append(errD.detach())

        # print('errD2plot', errD2plot)
        for i in range(len(errD2plot)):
            errDint.append(errD2plot[i].cpu().numpy())

        # list_ssim.append(int_ssim)
        # list_ssim_1.append(int_ssim_1)
        # list_ssim_2.append(int_ssim_2)
        # list_ssim_3.append(int_ssim_3)
        # list_ssim_4.append(int_ssim_4)
        # print('list_ssim', list_ssim)

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            output = netD(fake)
            # D_fake_map = output.detach()

            # errG均值函数
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha != 0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp * z_opt + z_prev
                rec_loss = alpha * loss(netG(Z_opt.detach(), z_prev), real)

                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()

        errGint = []
        errG2plot.append(errG.detach() + rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        for i in range(len(errG2plot)):
            errGint.append(errG2plot[i].cpu().numpy())

        if epoch % 100 == 0 or epoch == (opt.niter - 1):
            # len(Gs):scale   epoch= ,   niter = 2000
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            plt.imsave('%s/fake_sample.png' % (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png' % (opt.outf),
                       functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            # plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            plt.imsave('%s/noise.png' % (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            # plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)

            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

    # 目前是训练单次scale, 绘制第一次scale
    while (len(Gs) == 0):
        name = 'G_loos & G_loos'
        functions.plot_learning_curves(errGint, errDint,
                                       opt.niter, 'Generator', 'Discriminator', name)
        break

    while (len(Gs) == 1):
        name = 'G_loos & G_loos_1'
        functions.plot_learning_curves_1(errGint, errDint,
                                       opt.niter, 'Generator', 'Discriminator', name)
        break

    # while (len(Gs) == 2):
    #     name = 'G_loos & G_loos_2'
    #     functions.plot_learning_curves(errGint, errDint,
    #                                    list_ssim_3, list_ssim_4, opt.niter, 'labelG', 'labelD',
    #                                    'ssim_3', 'ssim_4', name)
    #     break

    # while (len(Gs) == 3):
    #     name = 'G_loos & G_loos_3'
    #     functions.plot_learning_curves(errGint, errDint, list_ssim, list_ssim_1, list_ssim_2,
    #                                    list_ssim_3, list_ssim_4, opt.niter, 'labelG', 'labelD', 'ssim',
    #                                    'ssim_1', 'ssim_2', 'ssim_3', 'ssim_4', name)
    #     break

    # while (len(Gs) == 4):
    #     name = 'G_loos & G_loos_4'
    #     functions.plot_learning_curves(errGint, errDint, list_ssim, list_ssim_1, list_ssim_2,
    #                                    list_ssim_3, list_ssim_4, opt.niter, 'labelG', 'labelD', 'ssim',
    #                                    'ssim_1', 'ssim_2', 'ssim_3', 'ssim_4', name)
    #     break
    #
    # while (len(Gs) == 5):
    #     name = 'G_loos & G_loos_5'
    #     functions.plot_learning_curves(errGint, errDint, list_ssim, list_ssim_1, list_ssim_2,
    #                                    list_ssim_3, list_ssim_4, opt.niter, 'labelG', 'labelD', 'ssim',
    #                                    'ssim_1', 'ssim_2', 'ssim_3', 'ssim_4', name)
    #     break
    #
    # while (len(Gs) == 6):
    #     name = 'G_loos & G_loos_6'
    #     functions.plot_learning_curves(errGint, errDint, list_ssim, list_ssim_1, list_ssim_2,
    #                                    list_ssim_3, list_ssim_4, opt.niter, 'labelG', 'labelD', 'ssim',
    #                                    'ssim_1', 'ssim_2', 'ssim_3', 'ssim_4', name)
    #     break
    #
    # while (len(Gs) == 7):
    #     name = 'G_loos & G_loos_7'
    #     functions.plot_learning_curves(errGint, errDint, list_ssim, list_ssim_1, list_ssim_2,
    #                                    list_ssim_3, list_ssim_4, opt.niter, 'labelG', 'labelD', 'ssim',
    #                                    'ssim_1', 'ssim_2', 'ssim_3', 'ssim_4', name)
    #     break
    #
    while (len(Gs) == 8):
        name = 'G_loos & G_loos_8'
        functions.plot_learning_curves_8(errGint, errDint,
                                       opt.niter, 'Generator', 'Discriminator', name)
        break

    functions.save_networks(netG, netD, z_opt, opt)

    return z_opt, in_s, netG


def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise])
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise(
                        [opt.nc_z, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise])
                z = m_noise(z)
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp * z + G_z
                G_z = G(z_in.detach(), G_z)
                G_z = imresize(G_z, 1 / opt.scale_factor, opt)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp * Z_opt + G_z
                G_z = G(z_in.detach(), G_z)
                G_z = imresize(G_z, 1 / opt.scale_factor, opt)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                # if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z


#
def init_models(opt):
    # 模型初始化
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
