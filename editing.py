from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images3D/')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Editing3D')
    parser.add_argument('--ref_name', help='reference image name', required=True)
    parser.add_argument('--editing_start_scale', help='editing injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='editing')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image3D(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.editing_start_scale < 1) | (opt.editing_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            ref = functions.read_image_dir3D('%s/%s' % (opt.ref_dir, opt.ref_name), opt) #1x1x40x40x40 for spyrals
            mask = functions.read_image_dir3D('%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-3],opt.ref_name[-3:]), opt)
            if ref.shape[3] != real.shape[3]:
                '''
                mask = imresize(mask, real.shape[3]/ref.shape[3], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize(ref, real.shape[3] / ref.shape[3], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]
                '''
                #print("\n \n DIFFERENT SHAPES\N\N")
                mask = imresize_to_shape(mask, [real.shape[2],real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2],real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            mask = functions.dilate_mask3D(mask, opt, debug=False)

            N = len(reals) - 1
            n = opt.editing_start_scale
            in_s = imresize3D(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[-3], :reals[n - 1].shape[-2], :reals[n - 1].shape[-1]]
            in_s = imresize3D(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[-3], :reals[n].shape[-2], :reals[n].shape[-1]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            torch.save(out, '%s/start_scale=%d.pt' % (dir2save, opt.editing_start_scale))
            out = (1-mask)*real+mask*out
            torch.save(out, '%s/start_scale=%d_masked.pt' % (dir2save, opt.editing_start_scale))




