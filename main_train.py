from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import shutil

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images3D/')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    copyCounter = 0

    tmpdir = dir2save
    while(os.path.exists(tmpdir)):
        try:
            #shutil.rmtree(dir2save)
            copyCounter += 1
            tmpdir = dir2save + "_" + str(copyCounter)
        except OSError:
            pass
    try:
        os.makedirs(tmpdir)
    except OSError:
        pass

    real = functions.read_image3D(opt)
    print(real.shape)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)