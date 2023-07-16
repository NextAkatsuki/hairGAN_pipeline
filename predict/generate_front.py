import os 
import cv2 
from use_model import Model_Run

class Generate():
    def __init__(self, usernum: str, img_name: str):
        self.img_name = img_name
        self.usernum = usernum
        # os.makedirs(f"../image/{usernum}/rawhair") #입력 이미지
        # os.makedirs(f"../image/{usernum}/dethair") #머리만 검출한 이미지
        # os.makedirs(f"../image/{usernum}/seghair") #정면이미지에서 검출된 머리만 가져온 이미지
        # os.makedirs(f"../image/{usernum}/ganhair") #검출된 머리를 이용하여 생성된 이미지
        # os.makedirs(f"../image/{usernum}/resulthair") #정면이미지와 생성된이미지를 합친 이미지(최종)


    #seg4paper의 정면이미지에서 머리인식용이미지를 이용하여 머리부분만 검출하여 hair4paper에 저장
    def _raw2seg(self, img, filename):
        a_path = f'../image/{self.usernum}/rawhair/' #raw
        # b_path = f'../image/{self.usernum}/dethair/' #seg
        f_path = f'../image/{self.usernum}/seghair/' #raw2seg

        a_image = cv2.imread(f'{a_path}{filename}')
        # b_image = cv2.imread(f'{b_path}{file_name}')

        gray_logo = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray_logo, 128, 255, cv2.THRESH_BINARY)

        for i in range(0,256):
            for j in range(0,mask_inv.shape[1]):
                if mask_inv[i][j] != 0:
                    a_image[i][j] = [255,255,255]

        cv2.imwrite(f'{f_path}{filename}', a_image)


    #정면 이미지(seg4paper)와 생성된 수술이후의 머리이미지를 합침
    def _seg2raw(self, img, file_name):
        a_path = f'../image/{self.usernum}/seghair/' #인풋
        # b_path = f'/data/CycleGAN-Tensorflow-2/output/{trans_model}/samples_testing/A2B/' #tran
        f_path = f"../image/{self.usernum}/resulthair" #최종결과(아웃풋)

        #s_image = cv2.imread(f'{s_path}{file_name}')
        #print(f'{s_path}{file_name}')
        a_image = cv2.imread(f'{a_path}{file_name}')
        # b_image = cv2.imread(f'{b_path}{file_name}')

        gray_logo = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray_logo, 128, 255, cv2.THRESH_BINARY)

        for i in range(0,256):
            for j in range(0,256):
                if mask_inv[i][j] == 0:
                    a_image[i][j] = img[i][j]

        cv2.imwrite(f'{f_path}{file_name}', a_image)

    def Get_Img_Dir(self):
        return f"../image/{self.usernum}/rawhair/{self.img_name}"

    def excute(self):
        input_img_path = os.path.join(f"image/{self.usernum}/rawhair", self.img_name)
        seg_image = Model_Run(experiment_dir="../output/seg4paper", input_img=input_img_path)
        self._raw2seg(seg_image, self.img_name)
        print("image being segmented")

        gan_image = Model_Run(experiment_dir="../output/hair4paper", input_img=input_img_path)
        self._seg2raw(gan_image, self.img_name)
        print("image being synthesized")

        print("done")

        return f"../image/{self.usernum}/resulthair/{self.img_name}"
    


# DEBUG
if __name__ == "__main__":
    test = Generate('1000','SU_00001_PRE_01.jpg')
    test.excute()


