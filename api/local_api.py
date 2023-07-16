import sys 
sys.path.append(r"/data")
from fastapi import FastAPI, UploadFile, status
import os 
import uuid
from pydantic import BaseModel
from predict.generate_front import Generate


app = FastAPI()

@app.get("/healthcheck", status_code=status.HTTP_200_OK)
async def healthcheck():
    return {"success": True}


@app.post("/imagetest/")
async def imagetest(file: UploadFile, userid: int):
    content = await file.read()
    filename = f"{str(uuid.uuid4())}.jpg"
    use_model = Generate(usernum=str(userid), img_name=filename)

    with open(use_model.Get_Img_Dir(),"wb") as fp:
        fp.write(content)

    try:
        result_img_dir = use_model.excute()
    except:
        return {"success": False, "message": "Model ERROR"}
    else:
        return {"success": True, "image": result_img_dir}
