import numpy as np
import AIC_scene_data
from torchvision import transforms
from AIC_scene_data import AIC_scene

def getEig():

    # facebook :
    eigValue = np.array([0.2175, 0.0188, 0.0045])
    eigVector = np.array([[-0.5675, 0.7192, 0.4009],
                          [-0.5808, -0.0045, -0.8140],
                          [-0.5836, -0.6948, 0.4203]])

    return eigValue, eigVector

if __name__ == "__main__":

    """
    meanStd = {"224": {'mean': [0.4951, 0.476, 0.4457],
                       'std': [0.2832, 0.2788, 0.2907]},
               "336": {'mean': [0.4952, 0.476, 0.4457],
                       'std': [0.2858, 0.2814, 0.2931]},
               "448": {'mean': [0.4927, 0.4735, 0.4435]}}

    train_data = AIC_scene(
        part="train",
        path="/data/chaoyang/scene_Classification",
        Transform=transforms.Compose([
            AIC_scene_data.Scale((224,224)),
            AIC_scene_data.CenterCrop(224),
            AIC_scene_data.ToTensor()
        ]))

    X = np.zeros((train_data.__len__(),3,224,224))
    for i in range(train_data.__len__()):
        X[i] = train_data[i]['image'].numpy()

    cov_01 = np.mean(np.dot(X[:,0,:,:]-meanStd['224']['mean'][0],
                            X[:,1,:,:]-meanStd['224']['mean'][1]))
    cov_02 = np.mean(np.dot(X[:,0,:,:]-meanStd['224']['mean'][0],
                            X[:,2,:,:]-meanStd['224']['mean'][2]))
    cov_12 = np.mean(np.dot(X[:,1,:,:]-meanStd['224']['mean'][1],
                            X[:,2,:,:]-meanStd['224']['mean'][2]))

    X = X.reshape((3,train_data.__len__()*224*224))
    cov = np.cov(X)
    eigValue,eigVec = np.linalg.eig(cov)
    """



