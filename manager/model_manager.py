from service_streamer import ManagedModel, Streamer
from fastapi.logger import logger
from .use_model import Generate_Model
import cv2
from functools import lru_cache

# ManagedModel 
class GenerateModelManager():
    #init_model
    def __init__(self):
        self.segmodel = Generate_Model(experiment_dir="volume/seg4paper",img_size=(256,256,3))
        self.hairmodel = Generate_Model(experiment_dir="volume/hair4paper",img_size=(256,256,3))


    #seg4paper의 정면이미지에서 머리인식용이미지를 이용하여 머리부분만 검출하여 hair4paper에 저장
    def _raw2seg(self, raw_imgs, seg_img, filename) -> list:
        result_dirs = []
        f_path = 'image/seghair/'
        gray_logo = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray_logo, 128, 255, cv2.THRESH_BINARY)

        for i in range(0,256):
            for j in range(0,mask_inv.shape[1]):
                if mask_inv[i][j] != 0:
                    raw_imgs[i][j] = [255,255,255]

        
        cv2.imwrite(f'{f_path}{filename}', raw_imgs)
        result_dirs.append(f'{f_path}{filename}')

        return result_dirs


    #정면 이미지(seg4paper)와 생성된 수술이후의 머리이미지를 합침
    def _seg2raw(self, raw_imgs, gan_img, filename) -> list:
        gray_logo = cv2.cvtColor(gan_img, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray_logo, 128, 255, cv2.THRESH_BINARY)

        for i in range(0,256):
            for j in range(0,256):
                if mask_inv[i][j] == 0:
                    raw_imgs[i][j] = gan_img[i][j]

        #debug
        # cv2.imwrite(f'{f_path}result.png', raw_imgs)
        return raw_imgs

    #inputs[0] = image_dirs:list inputs[1] = image_file_name:str
    def predict(self, inputs):
        logger.info(f"batch size: {len(inputs[0])}")
        result_imgs = []
        raw_seg_image = cv2.imread(inputs[0][0])
        raw_hair_image = cv2.imread(inputs[0][0])

        try:
            seg_image = self.segmodel.Model_Run(input_img_dirs=inputs[0], batch_size=len(inputs[0]))
            seg_img_dir = self._raw2seg(raw_imgs=raw_seg_image, seg_img = seg_image, filename=inputs[1])
            print("image being segmented")

            gan_image = self.hairmodel.Model_Run(input_img_dirs=seg_img_dir, batch_size=len(inputs[0]))
            gan_image = cv2.cvtColor(gan_image, cv2.COLOR_BGR2RGB)
            result_img = self._seg2raw(raw_imgs=raw_hair_image, gan_img = gan_image, filename=inputs[1])
            # cv2.imwrite("image/test.png",result_img)
            print("image being synthesized")
        except Exception as e:
            logger.error(f"Error {self.__class__.__name__}: {e}")
            print("Error to generate image")
            return []
        
        # result_imgs.append(result_img)
        print(result_img.shape)
        return result_img

#배치데이터부분에서 문제가 생김 (out of index)
@lru_cache(maxsize=32)
def get_model_generate_streamer():
    streamer = Streamer(
        GenerateModelManager,
        batch_size=8,
        max_latency=1,
        worker_num=4
    )
    return streamer

