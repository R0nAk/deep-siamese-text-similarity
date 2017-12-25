import os
import tensorflow as tf
from input_helpers import InputHelper
import numpy as np
from scipy import spatial

def getMetaFilePath(checkPointDir):
    meta_files = []
    inputPath = "runs/{0}/checkpoints/".format(checkPointDir)
    for file in os.listdir(inputPath):
        if file.endswith(".meta"):
            meta_files.append(file)
    sorted(meta_files, reverse=True)
    return inputPath+meta_files[0]

##############################################################################################################################
# under runs directory check latest checkpoint directory and get its .meta file
##############################################################################################################################
checkPoint = "1514142604"
metaFilePath = getMetaFilePath(checkPoint)
vocabFilePath = "runs/{0}/checkpoints/vocab".format(checkPoint)
lstmVectorFileName = "lstm_vector_output_{}".format(checkPoint)
batch_size = 64
#print(getMetaFilePath(checkPoint))
#print(vocabFilePath)

inputHelper = InputHelper()
###############################################################################################
#train_snli_with_qid - questionId_1,questionId_2,question_1,question_2,isDuplicate(0-no/1-yes)
###################################################################################################
qidOne,qidTwo,inputOne, inputTwo, target = inputHelper.getTestDataSetWithQuestIds("train_snli_with_qid.txt",vocabFilePath,30)


graph = tf.Graph()
questionIdList = []

with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph(metaFilePath)
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,metaFilePath[:-5])

        graph_input_one = graph.get_operation_by_name("input_x1").outputs[0]
        graph_input_two = graph.get_operation_by_name("input_x2").outputs[0]
        graph_input_target = graph.get_operation_by_name("input_y").outputs[0]
        graph_dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        outputOne = graph.get_operation_by_name("output/outputOne").outputs[0]
        outputTwo = graph.get_operation_by_name("output/outputTwo").outputs[0]

        batches = inputHelper.batch_iter(list(zip(inputOne, inputTwo, target,qidOne,qidTwo)), 2 *batch_size, 1, shuffle=False)
        counter = 0
        for batch in batches:
            print("Step {0} ".format(counter))
            counter+=1
            batchInputOne, batchInputTwo, batchOutput,batchQidOne,batchQidTwo = zip(*batch)
            outputOneTensorOne,outputTwoTensorTwo = sess.run([outputOne, outputTwo],{graph_input_one: batchInputOne, graph_input_two: batchInputTwo, graph_input_target:batchOutput, graph_dropout_keep_prob: 1.0})
            inputHelper.dumpLSTMTrainedOutputArray(outputOneTensorOne,outputTwoTensorTwo,lstmVectorFileName)
            inputHelper.dumpQuestionIds(batchQidOne,batchQidTwo)



##################################################################################################################################
# Once model is trained,All Question text gets converted into vector form and dump is taken in lstmVectorFileName file.
# For finding most similar questions, below code finds nearest neigbour(euclidian distance)
# Below code need to be replaced(using angular distance api-cosine)
##################################################################################################################################
print("Building KD tree now")
lstmVectorOutput = np.loadtxt(lstmVectorFileName,dtype=float)
tree = spatial.KDTree(lstmVectorOutput)
with open("lstm_similar.txt","a") as outFile:
    for i in xrange(lstmVectorOutput):
        indexList = tree.query(lstmVectorOutput[i],3)[1]
        similarQuestions = [questionIdList[index] for index in indexList]
        outFile.write("{0}  {1}\n".format(questionIdList[i],similarQuestions))



