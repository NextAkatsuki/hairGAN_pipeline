from fastapi import Depends
from fastapi.logger import logger


from manager import get_model_generate_streamer

class GenerateModel_Service():
    def __init__(
            self,
            generate_streamer=Depends(get_model_generate_streamer)
        ):
        logger.info(f"DI: {self.__class__.__name__}")
        self.generate_streamer = generate_streamer


    #예상) 이 함수를 요청에따라 계속호출하는듯 (image_dir이 단일갯수임)
    def predict_imgs(self, image_dir, k, image_name):
        result_img = self.generate_streamer.predict(([image_dir],image_name))
        return result_img