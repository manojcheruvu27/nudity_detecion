''' Open command prompt/Terminal and copy paste the following commands
    i) pip3 install --user --upgrade tensorflowh
    ii)python -m pip install nudenet --upgrade
    '''


"""nudity_detector.py: Provided with the path, this file is used to detect nudity in various
                       multimedia like images and videos."""

__author__  = "Sai Manoj Cheruvu"
__email__   = "manoj_cheruvu@outlook.com"

###### Credits: nudenet ######
####### Documentation : https://github.com/notAI-tech/NudeNet #######


##Import module
from nudenet import NudeDetector
from nudenet import NudeClassifier


##Initialize Nude Detector
detector = NudeDetector()
classifier = NudeClassifier()

def NudityDetector(path):
    
    '''Param1: Takes in the path of the image in which nudity has to be detected
       Returns a list containing information regarding which part is exposed and confidence levels:
       Example output: [{'box': [189, 275, 406, 474], 'score': 0.7914751768112183, 'label': 'EXPOSED_BREAST_F'},
       {'box': [408, 283, 614, 458], 'score': 0.7069545388221741, 'label': 'EXPOSED_BREAST_F'}]'''

    return detector.detect(path)



def NudityClassifier(path):
    '''Param1: Takes in the path of the image in which nudity is to be detected
       Returns a list containing the confidence intervals to what percent weather the picture is safe or unsafe
       Example output:  {'D:\\nudity_detection\\nudity.jpeg': {'safe': 0.027994073927402496, 'unsafe': 0.9720058441162109}}                                                                   '''

    return classifier.classify(path)


def CensorImage(path):
    '''Param1: Takes in the path of the image in which nudity has to be detected
       saves another image in which the nudity exposed parts are censored in the input image'''


    detector.censor(path, out_path='./image1_censored.jpg', visualize=False)

def batch_NudityClassifier(path,batchsize):
    '''param1: List of all the images to be classified
       param2: Batch size i.e the number of images to be classified
       Returns a list of confidence intervals with all the images classified as safe or unsafe
       Sample output : # {
       './image1.jpg': {
           'safe': 0.00015856922, 
           'unsafe': 0.99984145
     },
     './image2.jpg': {
         'safe': 0.019551795, 
         'unsafe': 0.9804482
     },
    './image3.jpg': {
         'safe': 0.00052562816,
         'unsafe': 0.99947435
    }, 
    './image4.jpg': {
         'safe': 3.3454136e-05,
         'unsafe': 0.9999665
    }
    }'''

    return classifier.classify(path,batch_size=batchsize)
def Nudity_in_video_classifier(path,batchsize):
    '''Param1: Path to the video in which the nudity needs to be classified
       param2: Batchsize generally equals to 4
       Returns {"metadata": {"fps": FPS, "video_length": TOTAL_N_FRAMES, "video_path": 'path_to_video'}
               "preds": {frame_i: {'safe': PROBABILITY, 'unsafe': PROBABILITY}, ....}}'''
    return classifier.classify_video('path_to_video', batch_size=batchsize)


##Driver code
if __name__ == '__main__':

    path = input("Enter the path of the image to be detected")
    print(NudityDetector(path))
    NudityClassifier(path)
    CensorImage(path)
    

