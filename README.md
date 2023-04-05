***作者：小成Charles
商业工作，学习交流请添加Vx：Lcc-Triumph
原创作品
转载请标注原创文章地址：*[https://blog.csdn.net/weixin_42999453/article/details/122755882](https://blog.csdn.net/weixin_42999453/article/details/122755261)
*本文代码下载地址Github:[https://github.com/xiaocheng99/IDRecognition.git](https://github.com/xiaocheng99/IDRecognition.git)**


## 一、前言
好久没有更新博客了，最近实习，接触了`OCR`的项目，感觉还挺有意思的，然后也发现了一款非常好用的`OCR`识别库，来自百度开发的`PaddleOCR`,识别率堪比商业级别。所以本文就没啥图像处理了，简单运用一下这个`PaddleOCR`。
## 二、PaddleOCR的使用
由于网上已经拥有安装和使用方法了，所以这里就不多赘述，给一下使用链接。
安装教程链接：[https://pythonmana.com/2022/01/202201030618069160.html](https://pythonmana.com/2022/01/202201030618069160.html)
使用教程链接：[https://aistudio.baidu.com/aistudio/projectdetail/507159?channelType=0&channel=0](https://aistudio.baidu.com/aistudio/projectdetail/507159?channelType=0&channel=0)
## 三、识别思路
接下来讲一下这个识别的思路，其实很简单，就是对获取的数据进行整合就OK。
**（1）直接对文字识别**

```python
    # 待预测图片
    test_img_path = [reversePath,frontPath]
    # 加载移动端预训练模型
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")

    np_images =[cv2.imread(image_path) for image_path in test_img_path] 

    #检测
    results = ocr.recognize_text(
                        images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                        use_gpu=True,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                        visualization=True,       # 是否将识别结果保存为图片文件；
                        box_thresh=0.5,           # 检测文本框置信度的阈值；
                        text_thresh=0.5)          # 识别中文文本置信度的阈值；
    #获取文字数据/
    resultStr = ''
    for result in results:
        data = result['data']
        save_path = result['save_path']
        for infomation in data:
            resultStr = resultStr+infomation['text']
```

上述代码块基本就是解决问题的核心，其中第一行的`reversePath`和`frontPath`分别是反面照片路径和正面的路径，然后通过加载训练模型，通过`ocr.recognize_text`函数即可扫描出图片中的文字数据，结果如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/74008e676e80487d848101377a94f190.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5bCP5oiQQ2hhcmxlcw==,size_17,color_FFFFFF,t_70,g_se,x_16)

然后们对获取的数据直接整合成一个长文本，即`resultStr`的结果为：

```bash
姓名充伊性别女民族汉出生1947年6月11日住址四川省成都市武侯区益州大道中段722号复城国际公民身份号码51370119470611660X居民身份证签发机关四川省成都市锦江分局有效期限2012.01.26-2032.01.21
```
**（2）对获取的数据进行筛选**
对数据提取之前我们要先删去那些我们不需要的东西 ，比如说可能出现的空格字符，因为`OCR`可能会误判从而多出来一些奇怪的符号

```python
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
```
**（3）对数据进行提取**
随后我们将拿到干净的数据，提取的思路就是我们找到这串数据中的`姓名`，`性别`，`出生`等关键词在文本中的序列号，然后进行分割再拼凑，具体实现如下，最后返回一个字典：

```python
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
```
到这里识别完成，最终识别的结果如下

```bash
{'姓名': '充伊', '性别': '女', '民族': '汉', '出生': '1947年6月11日', '住址': '四川省成都市武侯区益州大道中段722号复城国际', '
公民身份号码': '51370119470611660X', '签发机关': '四川省成都市锦江分局', '有效期开始时间': '2012.01.26', '有效期结束时间': '2032.01.21'}
```
**四、总结和注意事项**
本项目其实很简单，之前做这个项目一直用的图像识别技术去做，虽然也做出来了，但是识别精确度太低，直到发现这个宝藏OCR识别库。当然大家可能会碰到一些问题如下：

 1. 如果要使用`GPU`的话，第一步肯定得配置好`CUDA`的环境，先把`recognize`函数`GPU`改为`Ture`,然后在识别之前加上以下代码设置好GPU

```python
#设置、gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```
2.如果提示缺少`cudnn7.dll`之类的代码，就去百度找这个依赖库下载然后放到CUDA文件中的bin目录中

**作者：小成Charles
商业工作，学习交流请添加Vx：Lcc-Triumph
原创作品
转载请标注原创文章地址：[https://blog.csdn.net/weixin_42999453/article/details/122755882](https://blog.csdn.net/weixin_42999453/article/details/122755261)
*本文代码下载地址Github:[https://github.com/xiaocheng99/IDRecognition.git](https://github.com/xiaocheng99/IDRecognition.git)**
