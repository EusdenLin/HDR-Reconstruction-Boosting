import cv2
import os

def make_video(folder_path = '/disk2/twhuang/test_image/flower/', data = 'trajectory/', video_name = 'output_video.mp4'):
    image_path = os.path.join(folder_path, data)
    images = ['0.png', '-1.png', '-2.png', '-3.png']
    images.sort(key = lambda x:int(x[:-4]))
    frame = cv2.imread(os.path.join(image_path, images[-1]))
    print(os.path.join(image_path, images[-1]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(image_path, video_name), fourcc, 0.5, (width, height))

    print('Video writing')
    for image in images:
        print(cv2.imread(os.path.join(image_path, image)).shape)
        video.write(cv2.imread(os.path.join(image_path, image)))
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    cases = ['t60', 't68', 't78', 't91', 't95']
    for case in cases:
        make_video(folder_path='0926_gamma_inverse', data = case, video_name = case + 'video.mp4')
        for i in range(1, 4):
            make_video(folder_path=f'0926_gamma_inverse', data = case + f'/{i}_results', video_name = case + 'video.mp4')