import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import fmin
from tqdm import tqdm

class crf_est:
    def __init__(self, times=None, n_samples=0, sampling_strategy=None, smoothing_term=128, bNormalization=1):
        if n_samples < 1:
            self.n_samples = 256
        if times is None:
            raise ValueError("times must be specified")
        self.times = times
        if sampling_strategy is None:
            self.sampling_strategy = 'Grossberg'
        else:
            self.sampling_strategy = sampling_strategy
        self.smoothing_term = smoothing_term
        self.bNormalization = bNormalization

    def WeightFunction(self, img, weight_type='Deb97', bMeanWeight=0, bounds=[0, 1]):
        weight = None
        col = img.shape[0]
        
        if col > 1 and bMeanWeight:
            L = np.mean(img, axis=2)

            for i in range(img.shape[2]):
                img[:, :, i] = img[:, :, i] / L

        if weight_type == 'Deb97_p05':
            bounds = [0.05, 0.95]
            weight_type = 'Deb97'

        if weight_type == 'all':
            weight = np.ones(img.shape)
        elif weight_type == 'identity':
            weight = img
        elif weight_type == 'reverse':
            weight = 1 - img
        elif weight_type == 'box':
            weight = np.ones(img.shape)
            weight[img < bounds[0]] = 0
            weight[img < bounds[1]] = 0
        elif weight_type == 'hat':
            weight = 1 - (2 * img - 1) ** 12
        elif weight_type == 'Deb97':
            Zmin = bounds[0]
            Zmax = bounds[1]
            tr = (Zmin + Zmax) / 2
            delta = Zmax - Zmin

            indx1 = img <= tr
            indx2 = img > tr
            weight = np.zeros(img.shape)
            weight[indx1] = img[indx1] - Zmin
            weight[indx2] = Zmax - img[indx2]

            if delta > 0:
                weight = weight / tr
        else:
            raise ValueError("Weight type not recognized")

        weight[weight < 0] = 0
        weight[weight > 1] = 1

        return weight
    
    def LDRStackSubSampling(self, stack, outliers_percentage=0):
        if len(stack) == 0:
            raise ValueError("Stack is empty")

        def ComputeStackHistogram(stack):
            n, _, _, col = stack.shape
            stackOut = np.zeros((n, 256, col), dtype=np.float32)
            for i in range(n):
                for j in range(col):
                    tmp = stack[i, :, :, j]

                    if tmp.dtype in [np.float32, np.float64]:
                        tmp_conv = np.clip(np.round(tmp * 255), 0, 255).astype(np.uint8)
                    elif tmp.dtype == np.uint16:
                        tmp_conv = np.clip(np.round(tmp / 255), 0, 255).astype(np.uint8)
                        print("Warning: Is this a 16-bit image? The maximum is set to 65535.")
                    else:
                        # Assume the image is already in uint8 format
                        tmp_conv = tmp

                    hist, _ = np.histogram(tmp_conv, bins=256, range=(0, 256))
                    stackOut[i, :, j] = hist

            return stackOut
        
        def GrossbergSampling(stack):
            n, _, col = stack.shape
            for i in range(n):
                for j in range(col):
                    h_cdf = np.cumsum(stack[i, :, j])
                    stack[i, :, j] = h_cdf / h_cdf.max()
            
            delta = 1.0 / (self.n_samples - 1)
            u = np.arange(0, 1+delta, delta)
            stackOut = np.zeros((n, col, u.shape[0]), dtype=np.float32)
            
            # Vectorized implementation
            for k in range(n):
                for j in range(col):
                    # Compute all differences at once
                    diff_matrix = np.abs(stack[k, :, j].reshape(-1, 1) - u.reshape(1, -1))
                    # Get indices of minimum values along axis 0
                    stackOut[k, j, :] = np.argmin(diff_matrix, axis=0)
            
            return stackOut


        stack_hist = ComputeStackHistogram(stack)
        stack_samples = GrossbergSampling(stack_hist)

        return stack_samples
        

    def estimate(self, stack):        
        if stack is None or len(stack) == 0:
            raise ValueError("Stack is empty")
        
        stack = np.array(stack)
        
        if stack.dtype != np.float32:
            stack = stack.astype(np.float32)

        norm_stack = stack / 255.0
        weight = self.WeightFunction(np.arange(0, 1+1/255, 1/255), weight_type='Deb97')
        stack_samples = self.LDRStackSubSampling(stack)

        lin_fun = np.zeros((self.n_samples, 3), dtype=np.float32)
        log_stack_exposure = np.log(self.times)
        max_lin_fun = np.zeros((3), dtype=np.float32)

        def gsolve(Z, B, w, l):
            """
            Solve for the camera response function g.
            
            Parameters:
                Z : (rows, cols) numpy array of pixel values (expected in 0-255)
                B : 1D numpy array with exposure log values for each column of Z
                w : 1D numpy array of weights (length >= 256)
                l : smoothing constant (lambda)
            
            Returns:
                g : 1D numpy array of length 256, the estimated response curve.
            """
            cols, rows   = Z.shape
            n = 256
            m = rows * cols
            
            # Build the linear system: number of equations is m + n + 1, number of unknowns is n + rows.
            A = np.zeros((m + n + 1, n + rows), dtype=np.float32)
            b = np.zeros((A.shape[0], 1), dtype=np.float32)
            
            k = 0
            # Data-fitting term
            for i in range(rows):
                for j in range(cols):
                    # In MATLAB, the weight is computed by w(Z(i,j)+1)
                    # Adjust for Python's 0-indexing:
                    zij = int(Z[j, i])
                    wij = w[zij]
                    A[k, zij] = wij
                    A[k, n + i] = -wij
                    b[k, 0] = wij * B[j]
                    k += 1

            # Fix the curve by setting its middle value to 0.
            # MATLAB sets A(k,129)=1; → in Python index 128 when n==256.
            A[k, 128] = 1
            k += 1

            # Smoothness term
            for i in range(n - 2):
                A[k, i]   =  l * w[i+1]
                A[k, i+1] = -2 * l * w[i+1]
                A[k, i+2] =  l * w[i+1]
                k += 1

            # Solve the system using a least squares fit.
            # np.linalg.lstsq returns (x, residuals, rank, singular_values)
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            g = x[:n]
            return g.flatten()

        for i in range(3):
            g = gsolve(stack_samples[:, i, :], log_stack_exposure, weight, self.smoothing_term)
            g = np.exp(g)
            lin_fun[:, i] = g

        # color correction
        gray = np.zeros((3), dtype=np.float32)
        for i in range(3):
            gray[i] = np.mean(lin_fun[127, i])


        def FindChromaticyScale(M, I):
            """
            Find chromaticity scale factors that best match I to M.
            
            Parameters:
                M : array-like, reference color vector.
                I : array-like, input color vector.
                
            Returns:
                scale : 1D numpy array representing the chromaticity scale.
            """
            M = np.array(M, dtype=np.float32)
            I = np.array(I, dtype=np.float32).reshape(-1)
            
            l_m = M.shape[0]
            l_I = I.shape[0]
            
            if (l_m != l_I) or (M.size == 0) or (I.size == 0):
                raise ValueError("FindChromaticyScale: input colors have different color channels.")
            
            def residualFunction(p):
                I_c = I * p
                I_c_n = I_c / np.linalg.norm(I_c)
                M_n = M / np.linalg.norm(M)
                return np.sum((I_c_n - M_n)**2)
            
            # Initial guess: ones for each channel
            p0 = np.ones(l_m, dtype=np.float32)
            opts = {'xtol': 1e-12, 'ftol': 1e-12, 'disp': False}
            scale = fmin(residualFunction, p0, **opts)
            return scale

        scale = FindChromaticyScale([0.5, 0.5, 0.5], gray)
        for i in range(3):
            lin_fun[:, i] = lin_fun[:, i] * scale[i]
            max_lin_fun[i] = np.max(g)

        if self.bNormalization:
            max_val = np.max(max_lin_fun)
            for i in range(3):
                lin_fun[:, i] = lin_fun[:, i] / max_val

        return lin_fun
    
    def __call__(self, image):
        return self.estimate(image)

def readLDR(images_path):
      images = []
      norm_value = 255.0
      n = len(images_path)
      for i in range(n):
        #   print('Reading image: {}'.format(images_path[i]))
          img = cv2.resize(cv2.cvtColor(cv2.imread(images_path[i], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), (512, 512))
          img = img.astype(np.float32)
          img = img / norm_value
          images.append(img.copy())

      return images, norm_value

def process_case(args):
    case_path = args
    stack_path = []
    for i in range(-3, 1):
        stack_path.append(os.path.join(case_path, f"{i}.png"))
    
    stack, _ = readLDR(stack_path)
    crf_estimator = crf_est(times=times)
    output = crf_estimator.estimate(stack)
    savemat(os.path.join(case_path, 'response.mat'), {'lin_fun': output})
    return case_path

if __name__ == "__main__":
    # Example usage
    times = [8, 4, 2, 1]
    path = '/ssddisk/ytlin/data/HDR-Real/CEVR/'
    crf = crf_est(times=times)
    # for case in tqdm(os.listdir(path)):
    #     stack = []
    #     stack_path = []
    #     for i in range(-3, 1):
    #         stack_path.append(os.path.join(path, case, f"{i}.png"))

    #     stack, norm_value = readLDR(stack_path)
    #     output = crf.estimate(stack)
    #     savemat(os.path.join(path, case, 'response.mat'), {'lin_fun': output})

    case_paths = [os.path.join(path, case) for case in os.listdir(path)]
    
    with ProcessPoolExecutor(max_workers=64) as executor:
            results = list(tqdm(executor.map(process_case, case_paths), 
                            total=len(case_paths)))


    # # Save the output
    # for i in range(3):
    #     plt.plot(output[:, i], label=f'Channel {i}')
    # plt.title("Camera Response Function")
    # plt.xlabel("Pixel Value")
    # plt.ylabel("Response")
    # plt.legend()
    # plt.savefig("crf_estimation.png")
    