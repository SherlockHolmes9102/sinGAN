from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
import time

# python SR.py --input_dir /home/radiomics/djl/SinGAN-master/Input/Images --input_name single.png --sr_factor 4.0 --mode SR


if __name__ == '__main__':

    # 开始运行时间
    localtime_start = time.asctime(time.localtime(time.time()))
    # print(localtime_start)
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', default="86000_LR.png")  # required=True)
    parser.add_argument('--sr_factor', help='super resolution factor', type=float, default=4)
    parser.add_argument('--mode', help='task to be done', default='SR')
    opt = parser.parse_args()
    # 上传配置信息
    opt = functions.post_config(opt)

    Gs = []
    Zs = []
    reals = []
    # 噪声
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    # elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        mode = opt.mode
        # calc_init_scale返回了（尺度scale, 迭代次数）
        in_scale, iter_num = functions.calc_init_scale(opt)
        # 尺度因子
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale
        opt.mode = 'train'
        # 已经训练好的模型
        dir2trained_model = functions.generate_dir2save(opt)
        if (os.path.exists(dir2trained_model)):
            # 如果模型存在，加载load_trained_pyramid 生成的.pth文件
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            opt.mode = mode
        else:
            # 否则开始训练
            print('*** Train SinGAN for SR ***')
            real = functions.read_image(opt)
            functions.adjust_scales2image_SR(real, opt)
            train(opt, Gs, Zs, reals, NoiseAmp)
            opt.mode = mode
        print('%f' % pow(in_scale, iter_num))
        Zs_sr = []
        reals_sr = []
        NoiseAmp_sr = []
        Gs_sr = []
        real = reals[-1]  # read_image(opt)
        real_ = real
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale
        for j in range(1, iter_num + 1, 1):
            real_ = imresize(real_, pow(1 / opt.scale_factor, 1), opt)
            reals_sr.append(real_)
            Gs_sr.append(Gs[-1])
            NoiseAmp_sr.append(NoiseAmp[-1])
            z_opt = torch.full(real_.shape, 0, device=opt.device)
            m = nn.ZeroPad2d(5)
            z_opt = m(z_opt)
            Zs_sr.append(z_opt)
        out = SinGAN_generate(Gs_sr, Zs_sr, reals_sr, NoiseAmp_sr, opt, in_s=reals_sr[0], num_samples=1)
        out = out[:, :, 0:int(opt.sr_factor * reals[-1].shape[2]), 0:int(opt.sr_factor * reals[-1].shape[3])]
        dir2save = functions.generate_dir2save(opt)
        plt.imsave('%s/%s_HR.png' % (dir2save, opt.input_name[:-4]), functions.convert_image_np(out.detach()), vmin=0,
                   vmax=1)
        # 结束时间
        localtime_end = time.asctime(time.localtime(time.time()))
        print("开始时间：", localtime_start)
        print("结束时间：", localtime_end)
