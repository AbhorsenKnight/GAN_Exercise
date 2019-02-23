import numpy as np
import time
from skimage.transform import resize
from skimage.util.shape import view_as_windows

train_t = np.load('npy/BRATS2013_Syn_Flair_Train_X.npy')
train_y = np.load('npy/BRATS2013_Syn_Flair_Train_Y.npy')
test_t = np.load('npy/BRATS2013_Syn_Flair_Test_X.npy')
test_y = np.load('npy/BRATS2013_Syn_Flair_Test_Y.npy')

train_tt = train_t[:,1,:,:]
train_s = np.where(train_y==0)
train_t = train_tt[train_s[0],:,:]  #none leison brains
#print("train_tt size: ",train_tt.shape)
#print("train_t size: ",train_t.shape)
p_size = 21
image_size = 256

[z,y,x] = train_t.shape
cropw = 189                        # shape 200 = 10 + (10 + 160 + 10) + 10
startx = x//2 - (cropw//2)
starty = y//2 - (cropw//2)
train_tt = train_tt[:,starty:starty+cropw,startx:startx+cropw]

#print("train_patch size",train_t.shape)

test_t = test_t[:,1,:,:]
test_x = test_t[:,starty-10:starty+cropw+10,startx-10:startx+cropw+10]


train_t = view_as_windows(train_t,(1,p_size,p_size),step=3).reshape(-1,p_size,p_size)
#train_t = view_as_windows(train_t[:3000,:,:],(1,p_size,p_size)).reshape(-1,p_size,p_size)

#print("train_patch size",img_patch.shape)

train_empty = np.sum(np.sum(train_t,axis=1),axis=1)
train_notempty = np.where(train_empty!=0)
train_t = train_t[train_notempty[0],:,:]
train_x = resize(train_t[1:200000,:,:].transpose(1,2,0),(64,64)).transpose(2,0,1)
train_x = train_x.reshape(-1,64,64,1).astype(np.float32)


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "2"
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.models import load_model


import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=64, img_cols=64, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        

    #
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 32
        dropout = 0.4
        Alhpa = 0.2
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth = 32
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=1, input_shape=input_shape, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(MaxPooling2D(pool_size=(2,2), strides=None))
        self.D.add(LeakyReLU(alpha=Alhpa))
		    #self.D.add(ReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=1, padding='same'))
        self.D.add(BatchNormalization(momentum=0.9))
        self.D.add(MaxPooling2D(pool_size=(2,2), strides=None))
        self.D.add(LeakyReLU(alpha=Alhpa))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*3, 5, strides=1, padding='same'))
        self.D.add(MaxPooling2D(pool_size=(2,2), strides=None))
        self.D.add(LeakyReLU(alpha=Alhpa))
        self.D.add(Dropout(dropout))

        #self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        #self.D.add(LeakyReLU(alpha=0.2))
        #self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1024))
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 128
        dim = 8
        # In: 512
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=512))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))
		#
		# Out: 8*8*128
		

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth*2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
		
		#Out: 16*16*256

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth*2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
		
		#Out: 32*32*256
		
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 64 x 64 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 3, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        #optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        #optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.AM

class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channel = 1

        self.x_train = train_x
        #self.x_train = self.x_train.reshape(-1, self.img_rows,\
        #	self.img_cols, 1).astype(np.float32)

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        d_loss_all = np.zeros((train_steps,1))
        a_loss_all = np.zeros((train_steps,1))
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[64, 512])
        for i in range(train_steps):
            range_y = np.random.randint(0, self.x_train.shape[0], size=batch_size)
            images_train = self.x_train[range_y, :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 512])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.zeros([2*batch_size, 1])
            y[:batch_size, :] = 1
            d_loss = self.discriminator.train_on_batch(x, y)
            d_loss_all[i] = d_loss[0]

            y = np.ones([batch_size, 1]) 
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 512])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            a_loss_all[i] = a_loss[0]
            
            #print("D loss metric", self.discriminator.metrics_names)
            #print(d_loss[2],d_loss[3])
            
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))
                    file_gen_name = "model/hype_6/gen_%d" % (i+1)
                    file_dis_name = "model/hype_6/dis_%d" % (i+1)
                    file_adv_name = "model/hype_6/adv_%d" % (i+1)
                    #file_test_name = "test/test_%d.npy" % (i+1)
                    
                    self.generator.save(file_gen_name+'.h5')
                    self.discriminator.save(file_dis_name+'.h5')
                    self.adversarial.save(file_adv_name+'.h5')
                    
                    filename = "visual/hype_6/loss_%d.png" % (i+1)
                    
                    
                    plt.plot(d_loss_all[1:i])
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['d_loss'], loc='upper left')
                    plt.savefig(filename)
                    plt.close('all')
    
    def test(self):
        [l,y,x] = test_x.shape
        test_prob = np.zeros((l,189,189))
        for i in range(0,l):
            img_patch = view_as_windows(test_x[i,:,:], (p_size,p_size)).reshape(-1,p_size,p_size)
            img_patch = resize(img_patch.transpose(1,2,0),(64,64)).transpose(2,0,1).reshape(-1,64,64,1)
            img_predict = self.discriminator.predict(img_patch,batch_size=256)
            img_predict = img_predict.reshape(189,189)
            test_prob[i,:,:] = img_predict
            wordxx = "image %d test complete" % i
            print(wordxx)
            if (i+1) % 100 == 0:
                #file_test_name = "test/hype_6/test_p_%d.npy" % (i+1)
                file_test_name = "test/hype_6/test_p_%d.npy" % (i+1)
                np.save(file_test_name, test_prob, allow_pickle=False)  
        
        #file_test_final_name = "test/hype_6/test_final.npy"
        file_test_final_name = "test/hype_6/test_final.npy"     
        test_patch = test_prob
        np.save(file_test_final_name, test_patch, allow_pickle=False)
        
    def test_train(self):
        #[l,y,x] = test_x.shape
        file_test_train_name = "test/hype_4/test_train_tumor_indi.npy"  
        print("train_t size: ",train_tt.shape)
          
        range_t = np.random.randint(0,train_tt.shape[0], size=1000)
        np.save(file_test_train_name, range_t, allow_pickle=False) 
        test_t = train_tt[range_t,:,:]
        test_prob = np.zeros((test_t.shape[0],189,189))
        
        for i in range(0,1000):
            #print(test_t[i,:,:].shape)
            img_patch = view_as_windows(np.pad(test_t[i,:,:],((10,10),(10,10)),'edge'), (p_size,p_size)).reshape(-1,p_size,p_size)
            #print(img_patch.shape)
            img_patch = resize(img_patch.transpose(1,2,0),(64,64)).transpose(2,0,1).reshape(-1,64,64,1)
            #print(img_patch.shape)
            img_predict = self.discriminator.predict(img_patch,batch_size=256)
            img_predict = img_predict.reshape(189,189)
            test_prob[i,:,:] = img_predict
            wordxx = "image %d test complete" % i
            print(wordxx)
            if (i+1) % 100 == 0:
                file_test_name = "test/hype_4/test_a_t_%d.npy" % (i+1)
                np.save(file_test_name, test_prob, allow_pickle=False)  
        
        file_test_final_name = "test/hype_4/test_train_tumor_final_h4.npy"     
        test_patch = test_prob
        np.save(file_test_final_name, test_patch, allow_pickle=False)             
                                 

    def plot_images(self, save2file=False, fake=True, samples=64, noise=None, step=0):
        filename = 'visual/hype_6/brain.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 512])
            else:
                filename = "visual/hype_6/brain_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(8, 8, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()         
        

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    
    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # model = load_model('mymodel.h5')
    
    mnist_dcgan.discriminator = load_model('model/hype_4/dis_10000.h5')
    #mnist_dcgan.generator = load_model('model/hype_6/gen_10000.h5')
    #mnist_dcgan.adversarial = load_model('model/hype_6/adv_10000.h5')
    
    #mnist_dcgan.test()
    
    #mnist_dcgan.test()
    #mnist_dcgan.train(train_steps=60000, batch_size=256, save_interval=500)
    mnist_dcgan.test_train()
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)