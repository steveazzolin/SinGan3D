import SinGAN.functions as functions
import SinGAN.modelsCustom as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize3D
import SinGAN.customFuncs as customFuncs

def train(opt,Gs,Zs,reals,NoiseAmp):
    real3D_ = functions.read_image3D(opt)
    print("real")
    print(real3D_.shape)
    #real3D_ = customFuncs.get3D(real_)
    #real3D_ = customFuncs.genImage3Dv2((1,1,40,40,40))
    #real3D_ = customFuncs.genImageFunc((1,1,40,40,40))
    #real3D_ = customFuncs.genImageSpyral((1,1,40,40,40))
    #customFuncs.save3DFig(real3D_, "TrainedModels/piskelSmaller/_original.pt")
    real3D_ = functions.move_to_gpu(real3D_)
    print("real3D")
    print(real3D_.shape)
    in_s = 0
    scale_num = 0
    #real = imresize3D(real_,opt.scale1,opt)
    #print(real3D_)
    #customFuncs.visualizeVolume(real3D_)
    real3D = imresize3D(real3D_,opt.scale1,opt)
    #customFuncs.visualizeVolume(real3D)
    #print(real3D)
    #reals = functions.creat_reals_pyramid(real,reals,opt)
    reals3D = customFuncs.get3DPyramid(real3D,reals,opt)
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        #plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        D_curr,G_curr = init_models(opt)
        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))

        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals3D,Gs,Zs,in_s,NoiseAmp,opt)

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals3D, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return



def train_single_scale(netD,netG,reals3D,Gs,Zs,in_s,NoiseAmp,opt,centers=None):

    real = reals3D[len(Gs)]
    #print("REAL")
    #print(real.shape)
    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzz = real.shape[4]
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
        pad_noise = 0
    #m_noise = nn.ZeroPad2d(int(pad_noise)) #switch to 3d pad
    #m_image = nn.ZeroPad2d(int(pad_image)) #switch to 3d pad

    alpha = opt.alpha
    
    #fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device) #noise same shape of real
    fixed_noise3D = functions.generate_noise3D([1, opt.nzx, opt.nzy, opt.nzz], device=opt.device)
    #print(fixed_noise3D)
    #z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt3D = torch.full(fixed_noise3D.shape, 0, device=opt.device)
    #print(z_opt)
    #z_opt = m_noise(z_opt)
    z_opt3D = nn.functional.pad(z_opt3D, (5, 5, 5, 5, 5, 5), 'constant', 0)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter):
        if (Gs == []) & (opt.mode != 'SR_train'):
            #z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            z_opt3D = functions.generate_noise3D([1,opt.nzx,opt.nzy,opt.nzz], device=opt.device)
            #z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
            z_opt3D = nn.functional.pad(z_opt3D, (5, 5, 5, 5, 5, 5), 'constant', 0)
            #noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_3D = functions.generate_noise3D([1,opt.nzx,opt.nzy,opt.nzz], device=opt.device)
            #print("NOISEEEEEEEEE")
            #print(noise_3D.shape)
            #noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
            noise_3D = nn.functional.pad(noise_3D, (5, 5, 5, 5, 5, 5), 'constant', 0)
            #print("PADDED  NOISEEEEEEEEE")
            #print(noise_3D.shape)
            #print(noise_)
            #print(noise_.shape)
            #print(z_opt)
            #print(z_opt.shape)
        else:
            #noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_3D = functions.generate_noise3D([1,opt.nzx,opt.nzy,opt.nzz], device=opt.device)
            #noise_ = m_noise(noise_)
            noise_3D = nn.functional.pad(noise_3D, (5, 5, 5, 5, 5, 5), 'constant', 0)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()

            #print("Real")
            #print(real.shape)
            #print(real)
            #customFuncs.visualizeVolume(real)
            output = netD(real).to(opt.device)
            #D_real_map = output.detach()
            errD_real = -output.mean()#-a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            if (j==0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    #prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    prev = torch.full([1, 1, opt.nzx, opt.nzy, opt.nzz], 0, device=opt.device)
                    in_s = prev
                    #prev = m_image(prev)
                    prev = nn.functional.pad(prev, (5, 5, 5, 5, 5, 5), 'constant', 0)
                    #z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev3D = torch.full([1, 1, opt.nzx, opt.nzy, opt.nzz], 0, device=opt.device)
                    #z_prev = m_noise(z_prev)
                    z_prev3D = nn.functional.pad(z_prev3D, (5, 5, 5, 5, 5, 5), 'constant', 0)
                    opt.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev3D = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev3D))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev3D = m_image(z_prev3D)
                    prev = z_prev3D
                else:
                    prev = draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,'rand',opt)
                    #prev = m_image(prev)
                    prev = nn.functional.pad(prev, (5, 5, 5, 5, 5, 5), 'constant', 0)
                    z_prev3D = draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,'rec',opt) # RECONSTRUCTION LOSS
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev3D))
                    print(RMSE)
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    #z_prev = m_image(z_prev)
                    z_prev3D = nn.functional.pad(z_prev3D, (5, 5, 5, 5, 5, 5), 'constant', 0)
            else:
                prev = draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,'rand',opt)
                #prev = m_image(prev)
                prev = nn.functional.pad(prev, (5, 5, 5, 5, 5, 5), 'constant', 0)

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                #noise = noise_ # THERE IS NO IMAGE AT PREV LAYER THUS WE ONLY USE NOISE
                noise3D = noise_3D
            else:
                noise3D = opt.noise_amp*noise_3D+prev # HERE WE ADD NOISE TO PREV IMAGE

            #print("Noise")
            #print(noise3D.shape)
            #print(noise3D)
            #print("prev")
            #print(prev.shape)
            #customFuncs.visualizeVolume(noise3D)
            fake = netG(noise3D.detach(),prev) #error here
            #print("fake")
            #print(fake.shape)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            #print(gradient_penalty)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            output = netD(fake)
            #D_fake_map = output.detach()
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha!=0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp*z_opt3D+z_prev3D
                rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev3D),real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt3D
                rec_loss = 0

            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))
            print(errG)
            print(errD)

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            #plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt3D, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt3D,opt)
    #in_sl = torch.full(real.shape, 0, device=opt.device)
    #for i in range(0, 5):
        #z_currl = functions.generate_noise3D([1,opt.nzx,opt.nzy,opt.nzz], device=opt.device)
        #z_currl = z_currl.expand(1,1,z_currl.shape[2],z_currl.shape[3],z_currl.shape[4])
        #z_curr = m(z_curr)
        #z_currl = nn.functional.pad(z_currl, (5, 5, 5, 5, 5, 5), 'constant', 0)
        #I_prev = in_sl.clone()
        #I_prev = nn.functional.pad(I_prev, (5, 5, 5, 5, 5, 5), 'constant', 0)
        #z_in = opt.noise_amp*(z_currl)+I_prev
        #I_curr = netG(z_in.detach(),I_prev)
        #print(I_curr)
        #customFuncs.visualizeVolume(I_curr)
    return z_opt3D,in_s,netG  

def draw_concat3D(Gs,Zs,reals3D,NoiseAmp,in_s,mode,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals3D,reals3D[1:],NoiseAmp):
                if count == 0:
                    #z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z3D = functions.generate_noise3D([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise, Z_opt.shape[4] - 2 * pad_noise], device=opt.device)
                    #z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    #z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z3D = functions.generate_noise3D([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise, Z_opt.shape[4] - 2 * pad_noise], device=opt.device)
                #z = m_noise(z)
                z3D = nn.functional.pad(z3D, (5, 5, 5, 5, 5, 5), 'constant', 0)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3], 0:real_curr.shape[4]]
                #G_z = m_image(G_z)
                G_z = nn.functional.pad(G_z, (5, 5, 5, 5, 5, 5), 'constant', 0)
                z_in = noise_amp*z3D+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize3D(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3],0:real_next.shape[4]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals3D,reals3D[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3],0:real_next.shape[4]]
                #G_z = m_image(G_z)
                G_z = nn.functional.pad(G_z, (5, 5, 5, 5, 5, 5), 'constant', 0)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize3D(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3],0:real_next.shape[4]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z
    
    
# GENERATES UPSCALED IMAGE FOR CURRENT LEVEL STARTING FROM LEVEL 0 (BLANK IMAGE IN_S)
def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
