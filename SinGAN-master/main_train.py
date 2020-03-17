from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import time

# python main_train.py --input_dir /home/radiomics/djl/SinGAN-master/Input/Images --input_name 106mass.png --mode train

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')   # 生成随机样本
    opt = parser.parse_args()
    opt = functions.post_config(opt)

    # 生成列表
    Gs = []
    # 噪声列表？
    Zs = []
    # 图像列表
    reals = []

    # 噪声映射？
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    # 开始运行时间
    localtime_start = time.asctime(time.localtime(time.time()))

    if (os.path.exists(dir2save)):
        print('trained model already exist')

    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        # real读入真是数据
        real = functions.read_image(opt)

        functions.adjust_scales2image(real, opt)
        # 生成scale过程
        train(opt, Gs, Zs, reals, NoiseAmp)
        # 生成随机样本
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)

        # 结束时间
        localtime_end = time.asctime(time.localtime(time.time()))
        print("开始时间：", localtime_start)
        print("结束时间：", localtime_end)