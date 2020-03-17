# # main_train.opt
# #
# # Namespace(Dsteps=3, Gsteps=3, alpha=10, beta1=0.5, cuda=1, device=device(type='cuda', index=0),
# #                   fast_training=False, gamma=0.1, input_dir='/home/radiomics/djl/SinGAN-master/Input/Images',
# #                   input_name='single.png', ker_size=3, lambda_grad=0.1, lr_d=0.0005, lr_g=0.0005, manualSeed=1281,
# #                   max_size=256, min_nfc=32, min_nfc_init=32, min_size=25, mode='train', nc_im=3, nc_z=3, netD='',
# #                   netG='', nfc=32, nfc_init=32, niter=2000, niter_init=2000, noise_amp=0.1, noise_amp_init=0.1,
# #                   num_layer=5, out='Output', out_='TrainedModels/single/scale_factor=0.750000/', padd_size=0,
# #                   scale_factor=0.75, scale_factor_init=0.75, stride=1, workers=4)



# 0-3: 32
# 4-7:64
# 8-9:128


25
33
44
58
76
100
132
173
228
300

# 绘画损失列表
errD2plot = []
errG2plot = []
D_real2plot = []
D_fake2plot = []
z_opt2plot = []