from locust import HttpUser, task

class ServerTest(HttpUser):
    @task
    def api_test(self):
        with open('C:\\Users\\COMPUTER\\Desktop\\aihackathon\\hairGAN_pipeline/SU_00001_PRE_01.jpg','rb') as image:
            self.client.post("/predict",
                             files={'image':image})

