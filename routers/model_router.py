from fastapi_restful.inferring_router import InferringRouter
from fastapi.logger import logger
import uuid
import cv2
import numpy as np
from fastapi import (
    Depends,
    UploadFile,
    Response,
    Body
)

from fastapi_restful.cbv import cbv

from services import GenerateModel_Service
# from schema import PredictResult

router = InferringRouter()

@cbv(router)
class GenerateModel:
    svc: GenerateModel_Service = Depends()

    @router.post("/predict", responses = {
        200: {
            "content": {"image/jpg": {}}
        }
    },
        response_class= Response
    )
    def predict_img(
        self,
        image: UploadFile,
        k: int = Body(5, embed=True)
    ):
        logger.info("=======Predict Start=======")

        content = image.file.read()
        filename = f"{str(uuid.uuid4())}.jpg" #uuid로 유니크한 파일명으로 변경
        file_fullname = f"image/rawhair/{filename}"
        with open(file_fullname,"wb") as fp:
            fp.write(content) #서버 로컬 스토리지에 이미지 저장


        result_imgs = self.svc.predict_imgs(file_fullname, k, filename)
        _, encoded_image = cv2.imencode('.jpg',np.array(result_imgs))
        result_encod_img = encoded_image.tobytes()
        logger.info("=======Predict Done=======")

        return Response(content=result_encod_img, media_type="image/jpg")

