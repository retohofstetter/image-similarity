"""
Image similarity prediction with DeepNN
*******************************************
by ABIZ, RK, Reto Hofstetter
June 26, 2019

Used for calculating design similarity in:

Hofstetter, R., Nair, H., Misra, S. (2020), Can Open Innovation Survive? Imitation and Return on Originality in Crowdsourcing Creative Work. 
Stanford University Graduate School of Business Research Paper No. 18-11. Available at SSRN: https://ssrn.com/abstract=3133158

(Please cite if you like to reuse the code for your own project)
"""
print('---------- START CNN SIMILARITY CALCULATION -----------')
# In[] # import the required packages        
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3

class ImageSimilarity():
    
    
    if __name__ == '__main__':
    
        image1 = "C:/PATHTOYOURFILE1.JPG"
        image2 = "C:/PATHTOYOURFILE1.JPG"    
        
        target_size = (224, 224)
        input_shape = (target_size + (3,))
        batch_size = 1
        
        print("-> Loading InceptionV3 model...")    
        #eventually you will need to downgrade h5py in case of error loading the models
        #pip install "h5py==2.10.0" --force-reinstall
        model_name = InceptionV3
        model1 = model_name(weights='imagenet', include_top=False, input_shape=input_shape)
    
        print("-> Loading VGG16 model...")    
        model_name = VGG16
        model2 = model_name(weights='imagenet', include_top=False, input_shape=input_shape)
    
        print("-> Loading ResNet50 models...")    
        model_name = ResNet50
        model3 = model_name(weights='imagenet', include_top=False, input_shape=input_shape)
    
        print("-> Defining images...")    
        # initialize list of lists 
        data = [image1, image2] 
        df = pd.DataFrame(data, columns = ['filename']) 
    
        image_generator = ImageDataGenerator(#featurewise_center=True,
                                             #featurewise_std_normalization=True,
                                             #rotation_range=20,
                                             #width_shift_range=0.2,
                                             #height_shift_range=0.2,
                                             # horizontal_flip=True,
                                             rescale = 1.0/255.)
    
        img_array1 = img_to_array(load_img(image1))
        img_array2 = img_to_array(load_img(image2))
        img_arr = [img_array1, img_array2]
    
        image_generator.fit(img_arr)
        generated_images = image_generator.flow_from_dataframe(df, batch_size=batch_size, class_mode=None, target_size=target_size) 
    
        print("InceptionV3 Similarity (0: highly dissimilar, 1: highly similar)")
        features = model1.predict_generator(generated_images)
        feature_mat = []
        for i in range(len(features)):
            feature_mat.append(features[i].flatten())
        feature_arr = np.array(feature_mat)
        similarity_mat = cosine_similarity(feature_arr, feature_arr)
        inceptionv3 = similarity_mat[0][1]
        print(inceptionv3)
        
        print("VGG16 Similarity (0: highly dissimilar, 1: highly similar)")
        features = model2.predict_generator(generated_images)
        feature_mat = []
        for i in range(len(features)):
            feature_mat.append(features[i].flatten())
        feature_arr = np.array(feature_mat)
        similarity_mat = cosine_similarity(feature_arr, feature_arr)
        vgg16 = similarity_mat[0][1]
        print(vgg16)
        
        print("ResNet50 Similarity (0: highly dissimilar, 1: highly similar)")
        features = model3.predict_generator(generated_images)
        feature_mat = []
        for i in range(len(features)):
            feature_mat.append(features[i].flatten())
        feature_arr = np.array(feature_mat)
        similarity_mat = cosine_similarity(feature_arr, feature_arr)
        resnet50 = similarity_mat[0][1]
        print(resnet50)
    
        

         
            
            
            
   
