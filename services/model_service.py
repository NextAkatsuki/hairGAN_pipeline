from fastapi import Depends
from fastapi.logger import logger
import numpy as np


from manager import get_model_generate_streamer,GenerateModelManager


class GenerateModel_Service():
    def __init__(
            self,
            generate_streamer=Depends(get_model_generate_streamer)
        ):
        logger.info(f"DI: {self.__class__.__name__}")
        self.generate_streamer = generate_streamer
        self.model = GenerateModelManager()


    #예상) 이 함수를 요청에따라 계속호출하는듯 (image_dir이 단일갯수임)
    def predict_imgs(self, image_dir, k, image_name):
        # result_img = []
        # result_img.append(self.generate_streamer.predict(([image_dir],image_name)))
        # result_img = self.generate_streamer.predict(([image_dir],image_name))
        result_img = self.model.predict(([image_dir],image_name))
        print(np.array(result_img).shape)
        # cv2.imwrite("image/test.jpg",result_img)
        return result_img