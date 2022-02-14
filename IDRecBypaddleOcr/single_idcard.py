
from ast import arg
import time
import cv2
time1 = time.time()
import paddlehub as hub
import argparse
import sys
#设置、gpu
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def removeSpace (long_str):
    #去除空格
    noneSpaceStr = ''
    str_arry = long_str.split()
    for x in range(0,len(str_arry)):
        noneSpaceStr = noneSpaceStr+str_arry[x]
    return noneSpaceStr


def removePunctuation(noneSpaceStr):
   #去除标点符号
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！『【】（）、。：；’‘……￥·"""
    s =noneSpaceStr
    dicts={i:'' for i in punctuation}
    punc_table=str.maketrans(dicts)
    nonePunctuationStr=s.translate(punc_table)
    return nonePunctuationStr

def findResult(nonePunctuationStr):
    name = "姓名"
    sex = "性别"
    race = "民族"
    birth = "出生"
    address = "住址"
    idCardNumber = "公民身份号码"
    issuedBy = '签发机关'
    validDate = '有效期限'
    validDateStart = '有效期开始时间'
    validDateEnd = '有效期结束时间'

    indexName = nonePunctuationStr.find(name)
    indexSex = nonePunctuationStr.find(sex)
    indexRace = nonePunctuationStr.find(race)
    indexBirth = nonePunctuationStr.find(birth)
    indexAddress = nonePunctuationStr.find(address)
    indexIdCardNumber = nonePunctuationStr.find(idCardNumber)
    indexIssuedBy = nonePunctuationStr.find(issuedBy)
    indexValidDate = nonePunctuationStr.find(validDate)


    
    numberName = nonePunctuationStr[indexName+2:indexSex]
    numberSex = nonePunctuationStr[indexSex+2:indexSex+3]
    numberRace = nonePunctuationStr[indexRace+2:indexRace+3]
    numberBirth = nonePunctuationStr[indexBirth+2:indexAddress]
    numberAddress = nonePunctuationStr[indexAddress+2:indexIdCardNumber]
    numberIdCardNumber = nonePunctuationStr[indexIdCardNumber+6:indexIdCardNumber+24]
    strIssuedBy = nonePunctuationStr[indexIssuedBy+4:indexValidDate]
    strDate = nonePunctuationStr[indexValidDate+4:len(nonePunctuationStr)]
    strValidDateStart = strDate[0:4]+"."+strDate[4:6]+"."+strDate[6:8]
    strValidDateEnd = strDate[8:12]+"."+strDate[12:14]+"."+strDate[14:16]

    reverseDict = {name:numberName,sex:numberSex,race:numberRace,birth:numberBirth,address:numberAddress,idCardNumber:numberIdCardNumber,issuedBy:strIssuedBy,validDateStart:strValidDateStart,validDateEnd:strValidDateEnd}
    return reverseDict


    
def getInformation(resultStr):

    # #去除空格
    NoneSpaceStr = removeSpace(resultStr)

    # #去除标点符号
    NonePunctuationStr = removePunctuation(NoneSpaceStr)

    #数据提取
    resultDict = findResult(NonePunctuationStr)
  
    #合并字典返回
    return dict(resultDict)

    ########身份证反面信息识别
def identity_OCR(reversePath,frontPath):
    # 待预测图片
    test_img_path = [reversePath,frontPath]
    # 加载移动端预训练模型
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")

    np_images =[cv2.imread(image_path) for image_path in test_img_path] 

    #检测
    results = ocr.recognize_text(
                        images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                        use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                        visualization=False,       # 是否将识别结果保存为图片文件；
                        box_thresh=0.5,           # 检测文本框置信度的阈值；
                        text_thresh=0.5)          # 识别中文文本置信度的阈值；
    #获取文字数据/
    resultStr = ''
    for result in results:
        data = result['data']
        save_path = result['save_path']
        for infomation in data:
            resultStr = resultStr+infomation['text']
    #print(resultStr)

    #进行数据提取 返回字典
    mregeDict =getInformation(resultStr)
    return mregeDict


if __name__ == '__main__':
    # # 反面（人像）图片
    # reversePath = './1.png'
    # # 正面（国徽）图片  
    # frontPath = './2.jpg'

    parser = argparse.ArgumentParser()
    parser.add_argument('--reverse',type=str,help='The path of reverse side which has the picture of people')
    parser.add_argument('--front',type=str,help='The path of front side which has the Coat of arms of China')
    args = parser.parse_args()
    if not args.reverse or not args.front:
        parser.print_help()
        sys.exit(0)
    reversePath = str(args.reverse)
    frontPath = str(args.front)
    mregeDict = identity_OCR(reversePath,frontPath)
    print(mregeDict)

