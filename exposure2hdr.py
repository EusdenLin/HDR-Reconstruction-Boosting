# covnert exposure bracket to HDR output
import argparse 
import os 
import cv2
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import skimage
import ezexr
from relighting.tonemapper import TonemapHDR

def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help='directory that contain the image') #dataset name or directory 
    parser.add_argument("--output_dir", type=str, required=True, help='directory that contain the image') #dataset name or directory 
    parser.add_argument("--endwith", type=str, default=".png" ,help='file ending to filter out unwant image')
    parser.add_argument("--ev_string", type=str, default="_ev" ,help='string that use for search ev value')
    parser.add_argument("--EV", type=str, default="0, -1, -2, -3" ,help='avalible ev value')
    parser.add_argument("--iteration", default='', help="iteration", type=str)
    parser.add_argument("--gamma", default=2.2, help="Gamma value", type=float)
    parser.add_argument('--preview_output', dest='preview_output', action='store_true')
    parser.set_defaults(preview_output=True)
    return parser

def parse_filename(ev_string, endwith,filename):
    a = filename.split(ev_string)
    name = ev_string.join(a[:-1])
    ev = a[-1].replace(endwith, "")
    ev = int(ev)
    return {
        'name': name,
        'ev': ev,
        'filename': filename
    }

def process_image(args, info):
    
    #output directory
    hdrdir = args.output_dir
    os.makedirs(hdrdir, exist_ok=True)
    
    scaler = np.array([0.212671, 0.715160, 0.072169])
    name = info['name']
    # ev value for each file
    evs = [e for e in sorted(info['ev'], reverse = True)]

    # filename
    files = [info['ev'][e] for e in evs]

    # weights = []
    # for ev in evs:
    #     weight = 1 / (2 ** abs(ev))
    #     weights.append(weight)
    # weights = np.array(weights)
    # weights /= weights.sum()  # Normalize weights
    # print(evs, weights)

    # images = []
    # for file in files:
    #     path = os.path.join(args.input_dir, file)
    #     img = skimage.io.imread(path)[...,:3]
    #     img = skimage.img_as_float(img)
    #     linear_img = np.power(img, args.gamma)
    #     linear_img *= 1 / (2 ** evs[files.index(file)])

    #     images.append(linear_img)

    # hdr_rgb = np.zeros_like(images[0], dtype=np.float32)
    # for img, weight in zip(images, weights):
    #     hdr_rgb += img * weight
    
    # inital first image
    image0 = skimage.io.imread(os.path.join(args.input_dir, files[0]))[...,:3]
    image0 = skimage.img_as_float(image0)
    image0_linear = np.power(image0, args.gamma)

    # read luminace for every image 
    luminances = []
    for i in range(len(evs)):
        # load image
        path = os.path.join(args.input_dir, files[i])
        image = skimage.io.imread(path)[...,:3]
        image = skimage.img_as_float(image)
        
        # apply gama correction
        linear_img = np.power(image, args.gamma)
        
        # convert the brighness
        linear_img *= 1 / (2 ** evs[i])
        
        # compute luminace
        lumi = linear_img @ scaler
        # breakpoint()
        luminances.append(linear_img)
        
    # print(luminances[0].shape, len(luminances))
    # start from darkest image
    out_luminace = luminances[len(evs) - 1]

    # for i in range(len(evs) - 1, 0, -1):
    #     # compute mask
    #     maxval = 1 / (2 ** evs[i-1])
    #     p1 = np.clip((luminances[i-1] - 0.9 * maxval) / (0.1 * maxval), 0, 1)
    #     p2 = out_luminace > luminances[i-1]
    #     mask = (p1 * p2).astype(np.float32)
    #     # out_luminace = luminances[i-1] * (1-mask) + out_luminace * mask
    #     out_luminace = out_luminace
        
    hdr_rgb = image0_linear * (out_luminace / (luminances[0] + 1e-10))
    
    # print(hdr_rgb.shape)
    # tone map for visualization    
    hdr2ldr = TonemapHDR(gamma=args.gamma, percentile=99, max_mapping=0.9)
    
    
    ldr_rgb, _, _ = hdr2ldr(hdr_rgb)
    
    if args.preview_output:
        preview_dir = os.path.join(args.output_dir, args.iteration + "_tone_mapped")
        os.makedirs(preview_dir, exist_ok=True)

        skimage.io.imsave(os.path.join(preview_dir, "tonemapped.png"), skimage.img_as_ubyte(ldr_rgb))
        ezexr.imwrite(os.path.join(preview_dir, "hdr.exr"), hdr_rgb.astype(np.float32))
        
        bracket = []
        for s, num in zip(2 ** np.linspace(0, evs[-1], 4), np.linspace(0, evs[-1], 4)): #evs[-1] is -5
            lumi = np.clip((s * hdr_rgb) ** (1/args.gamma), 0, 1)
            skimage.io.imsave(os.path.join(preview_dir, f"{int(num)}.png"), skimage.img_as_ubyte(lumi))
            bracket.append(lumi)
        bracket = np.concatenate(bracket, axis=1)
        # skimage.io.imsave(os.path.join(preview_dir, name+".png"), skimage.img_as_ubyte(bracket))
    return None

def main():
    # load arguments
    args = create_argparser().parse_args()
    
    files = sorted(os.listdir(args.input_dir))
    
    #filter file out with file ending
    files = [f for f in files if f.endswith(args.endwith)]
    evs = [float(e.strip()) for e in args.EV.split(",")]
    
    # parse into useful data
    files = [parse_filename(args.ev_string, args.endwith, f) for f in files]
    
    # filter out unused ev
    files = [f for f in files if f['ev'] in evs] 
  
    info = {}
    for f in files:
        if not f['name'] in info:
            info[f['name']] = {}
        info[f['name']][f['ev']] = f['filename']

    infolist = []
    for k in info:
        if len(info[k]) != len(evs):
            print("WARNING: missing ev in ", k)
            continue
        # convert to list data
        print(k, info[k])
        infolist.append({'name': k, 'ev': info[k]})

    process_image(args, infolist[0])
    
    fn = partial(process_image, args)
    with Pool(8) as p:
        r = list(tqdm(p.imap(fn, infolist), total=len(infolist)))
    
    print("Done")

     
    
if __name__ == "__main__":
    main()