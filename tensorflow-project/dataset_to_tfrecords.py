import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile

OUTPUT_FILE1 = '/home/sever/桌面/CUB_200_2011/train_data.npy'
OUTPUT_FILE2 = '/home/sever/桌面/CUB_200_2011/test_data.npy'
#OUTPUT_FILE3= '/home/sever/桌面/CUB_200_2011/test_images.npy'
#OUTPUT_FILE4= '/home/sever/桌面/CUB_200_2011/test_labels.npy'

def load_image_labels(dataset_path=''):
  labels = {}
  
  with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      class_id = pieces[1]
      labels[image_id] = int(class_id)
  return labels


def load_train_test_split(dataset_path=''):
  train_images_id = []
  test_images_id = []
  
  with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      is_train = int(pieces[1])
      if is_train > 0:
        train_images_id.append(image_id)
      else:
        test_images_id.append(image_id)
        
  return train_images_id, test_images_id

def load_image_path(dataset_path=''):
  image_paths = {}
  with open(os.path.join(dataset_path,'images.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      path = os.path.join(dataset_path,'images',pieces[1])
      image_paths[image_id] = path
  return image_paths

def load_image(dataset_path=''):
  image_raw_data = gfile.FastGFile(dataset_path, 'rb').read()
  image = tf.image.decode_jpeg(image_raw_data)
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [200, 200])
    image_raw = image.tobytes()
    return image_raw
  
    
labels = load_image_labels(dataset_path='/home/sever/桌面/CUB_200_2011')
#print(labels)
train_images_id,test_images_id = load_train_test_split(dataset_path = '/home/sever/桌面/CUB_200_2011')
#print(train_images_id)
#print(test_images_id)
image_paths = load_image_path(dataset_path='/home/sever/桌面/CUB_200_2011')



for image_id in train_images_id:
  train_labels = (labels[image_id])
  file_name = image_paths[image_id]
  train_image_raw = load_image(dataset_path = file_name)
  example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[train_labels])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字
  print(image_id,'train_image processed')
writer.close()
  #print(np.array(train_images))
'''print(labels)
print(train_images[20])
print(train_labels[20])
plt.imshow(train_images[20])
plt.show()'''
#print(np.array(train_labels))

#for image_id in test_images_id:
  #test_labels.append(labels[image_id])
  #file_name = image_paths[image_id]
  #test_image_value = load_image(sess,dataset_path = file_name)
  #test_images.append(test_image_value)
  #print(image_id,'test_image processed')

#train_data = np.asarray([train_images,train_labels])
#test_data = np.asarray([test_images,test_labels])


#np.save(OUTPUT_FILE1,train_data)
#np.save(OUTPUT_FILE2,test_data)

'''def read_and_decode(filename): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])  #reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    return img, label'''

  
