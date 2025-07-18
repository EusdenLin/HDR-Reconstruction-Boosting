import numpy as np
import ezexr
import cv2
import os

def ReinhardTMO(img_hdr, key=0.18):
    """
    Apply Reinhard tone mapping operator to an HDR image.
    
    Parameters:
        img_hdr: HDR image array
        key: Key value (controls brightness, default 0.18)
    
    Returns:
        Tone-mapped image
    """
    # If the image has 3 channels, compute luminance
    if len(img_hdr.shape) == 3 and img_hdr.shape[2] == 3:
        # Convert to luminance using standard RGB weights
        luminance = 0.212671 * img_hdr[:,:,0] + 0.715160 * img_hdr[:,:,1] + 0.072169 * img_hdr[:,:,2]
        luminance = np.maximum(luminance, 1e-10)  # Avoid division by zero
        
        # Calculate the scaled luminance
        L_w_avg = np.exp(np.mean(np.log(luminance + 1e-10)))  # Log-average luminance
        L = key * luminance / L_w_avg
        
        # Apply Reinhard operator
        L_d = L / (1.0 + L)
        
        # Scale the RGB channels by ratio of tone-mapped to original luminance
        ratio = L_d / luminance
        img_tmo = np.zeros_like(img_hdr)
        for c in range(3):
            img_tmo[:,:,c] = img_hdr[:,:,c] * ratio
    else:
        # Single channel image (assume it's already luminance)
        L_w_avg = np.exp(np.mean(np.log(img_hdr + 1e-10)))
        L = key * img_hdr / L_w_avg
        img_tmo = L / (1.0 + L)
    
    return img_tmo

def KimKautzTMO(img_hdr, Ld_max=300, Ld_min=0.3, KK_c1=3.0, KK_c2=0.5):
    """
    Apply Kim-Kautz Consistent Tone Mapping Operator to an HDR image.
    
    Parameters:
        img_hdr: HDR image array
        Ld_max: Max luminance of the LDR monitor in cd/m^2 (default: 300)
        Ld_min: Min luminance of the LDR monitor in cd/m^2 (default: 0.3)
        KK_c1: Parameter adjusting the shape of Gaussian fall-off (default: 3.0)
        KK_c2: Dynamic range ratio parameter (default: 0.5)
    
    Returns:
        Tone-mapped image
    """
    # Ensure no negative values
    img_hdr = np.maximum(img_hdr, 0.0)
    
    # Calculate luminance
    if len(img_hdr.shape) == 3 and img_hdr.shape[2] == 3:
        L = 0.212671 * img_hdr[:,:,0] + 0.715160 * img_hdr[:,:,1] + 0.072169 * img_hdr[:,:,2]
    else:
        L = img_hdr.copy()  # Single channel image
    
    # Log luminance
    L_log = np.log(L + 1e-6)
    
    # Calculate mean log luminance
    mu = np.mean(L_log)
    
    # Calculate min and max log luminance
    maxL = np.max(L_log)
    minL = np.min(L_log)
    
    # Calculate min and max display log luminance
    maxLd = np.log(Ld_max)
    minLd = np.log(Ld_min)
    
    # Calculate k1 parameter
    k1 = (maxLd - minLd) / (maxL - minL)
    
    # Calculate dynamic range and sigma
    d0 = maxL - minL
    sigma = d0 / KK_c1
    
    # Calculate weight using Gaussian
    sigma_sq_2 = (sigma**2) * 2
    w = np.exp(-((L_log - mu)**2) / sigma_sq_2)
    
    # Calculate k2 parameter
    k2 = (1 - k1) * w + k1
    
    # Apply tone mapping formula
    Ld = np.exp(KK_c2 * k2 * (L_log - mu) + mu)
    
    # Robust min and max (using percentiles instead of MaxQuart)
    minLd = np.percentile(Ld, 1)
    maxLd = np.percentile(Ld, 99)
    
    # Clamp values
    Ld = np.clip(Ld, minLd, maxLd)
    
    # Normalize to [0,1]
    Ld = (Ld - minLd) / (maxLd - minLd)
    
    # Change luminance while preserving colors
    img_out = np.zeros_like(img_hdr)
    if len(img_hdr.shape) == 3 and img_hdr.shape[2] == 3:
        # For each color channel, scale by ratio of new/old luminance
        ratio = np.divide(Ld, L, where=L>0)
        for c in range(3):
            img_out[:,:,c] = img_hdr[:,:,c] * ratio
    else:
        img_out = Ld  # Single channel image
    
    # Remove special values (NaN, Inf)
    img_out = np.nan_to_num(img_out)
    
    return img_out

def GammaTMO(img, TMO_gamma, TMO_fstop, TMO_view):
    """
    Apply gamma correction and exposure adjustment to an image.
    
    Parameters:
        img: Input image
        TMO_gamma: Gamma value (e.g., 2.2)
        TMO_fstop: Exposure adjustment in f-stops (positive values brighten, negative darken)
        TMO_view: View parameter (typically 1.0 for standard view)
    
    Returns:
        Gamma-corrected and exposure-adjusted image
    """
    # Apply exposure adjustment (f-stop)
    # 2^(f-stop) adjusts exposure, positive values brighten, negative darken
    exposure_factor = 2.0 ** TMO_fstop
    img_exposed = img * exposure_factor * TMO_view
    
    # Clip to [0, 1] range
    img_exposed = np.clip(img_exposed, 0, 1)
    
    # Apply gamma correction
    img_gamma = np.power(img_exposed, 1.0 / TMO_gamma)
    
    return img_gamma

def tone_map_hdr(img_hdr, key=0.18, gamma=2.2, fstop=0, view=1, tmo='reinhard'):
    """
    Combined function to apply Reinhard TMO followed by gamma correction.
    
    Parameters:
        img_hdr: HDR image array
        key: Key value for Reinhard TMO
        gamma: Gamma value
        fstop: Exposure adjustment in f-stops
        view: View parameter
        tmo: Tone mapping operator ('reinhard' or 'gamma' or )

    Returns:
        Tone-mapped image
    """
    tmoImg = None
    if tmo == 'reinhard':
        tmoImg = ReinhardTMO(img_hdr, key)
    elif tmo == 'gamma':
        tmoImg = img_hdr
    elif tmo == 'kimkautz':
        tmoImg = KimKautzTMO(img_hdr)
    else:
        raise ValueError(f"Unknown TMO: {tmo}")
    
    return GammaTMO(tmoImg, gamma, fstop, view)

if __name__ == "__main__":
    remake = []
    # path = '/ssddisk/ytlin/results_clip/intermediate_HDR-Real/single_boost/01777/4_tone_mapped/hdr.exr'
    path = '/home/ytlin/boosting_HDR/results_clip/VDS/Deep_Recursive_HDRI'
    output_path = '/home/ytlin/boosting_HDR/results_clip/VDS/RH_TMO/Deep_Recursive_HDRI'
    # case = ['00806']
    for file in os.listdir(path):
    # for file in case:
        print(file)
        if not os.path.exists(os.path.join(path, file, 'inpaint/hdr.hdr')):
            remake.append(file)
            continue
        os.makedirs(os.path.join(output_path, file), exist_ok=True)
        hdr = cv2.imread(os.path.join(path, file, 'inpaint/hdr.hdr'), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH) 
        # cv2.imwrite(os.path.join(output_path, file, 'hdr.exr'), hdr)
        hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
        ldr = tone_map_hdr(hdr, key=0.18, gamma=2.2, fstop=0, view=1, tmo='reinhard')
        ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, file, 'inpaint.png'), np.clip((ldr * 255), 0, 255).astype(np.uint8))
    output_path = '/home/ytlin/boosting_HDR/results_clip/VDS/KK_TMO/Deep_Recursive_HDRI'
    for file in os.listdir(path):
    # for file in case:
        print(file)
        if not os.path.exists(os.path.join(path, file, 'inpaint/hdr.hdr')):
            remake.append(file)
            continue
        os.makedirs(os.path.join(output_path, file), exist_ok=True)
        hdr = cv2.imread(os.path.join(path, file, 'inpaint/hdr.hdr'), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH) 
        # cv2.imwrite(os.path.join(output_path, file, 'inpaint.png'), np.clip((hdr * 10), 0, 255).astype(np.uint8))
        # cv2.imwrite(os.path.join(output_path, file, 'hdr.exr'), hdr)
        hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
        ldr = tone_map_hdr(hdr, key=0.18, gamma=2.2, fstop=0, view=1, tmo='kimkautz')
        ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, file, 'inpaint.png'), np.clip((ldr * 255), 0, 255).astype(np.uint8))
    # hdr = ezexr.imread(path)
    # hdr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB) 
    # ldr = tone_map_hdr(hdr, key=0.18, gamma=2.2, fstop=0, view=1, tmo='reinhard')

    # # cv2.imwrite('./tonemapped.png', ldr * 255)
    # cv2.imwrite('./tonemapped.png', ldr * 255)

    print(remake)
