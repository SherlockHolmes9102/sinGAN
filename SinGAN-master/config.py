import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', help='task to be done', default='train')
    # workspace:
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=1)

    # 加载输入保存配置:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z', type=int, help='noise # channels', default=3)
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)
    parser.add_argument('--out', help='output folder', default='Output')

    # 网络超参数:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    parser.add_argument('--num_layer', type=int, help='number of layers', default=5)
    parser.add_argument('--stride', help='stride', default=1)
    parser.add_argument('--padd_size', type=int, help='net pad size', default=0)  # math.floor(opt.ker_size/2)

    # 金字塔参数:
    # scale_factor  尺度因子
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor', default=0.75)  # pow(0.5,1/6))
    parser.add_argument('--noise_amp', type=float, help='addative noise cont weight', default=0.1)
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=25)
    # parser.add_argument('--max_size', type=int, help='image minimal size at the coarser scale', default=256)
    parser.add_argument('--max_size', type=int, help='image minimal size at the coarser scale', default=300)
    # 优化器参数:

    # niter迭代次数
    # parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train per scale')
    parser.add_argument('--fast_training', type=bool, help='fast training flag', default=False)
    parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=3)
    parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=3)
    # 梯度惩罚权重
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
    # 重构损失权重
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10)

    return parser
