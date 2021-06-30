
import cv2
import numpy as np
 
'''
retval = cv.createBackgroundSubtractorMOG2( [, history[, varThreshold[, detectShadows]]] )
Parameters
    history Length of the history.
    varThreshold    Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model. This parameter does not affect the background update.
    detectShadows   If true, the algorithm will detect shadows and mark them. It decreases the speed a bit, so if you do not need this feature, set the parameter to false.
url: https://docs.opencv.org/3.4.2/de/de1/group__video__motion.html#ga2beb2dee7a073809ccec60f145b6b29c
'''
#MOG背景分割器
mog = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
camera = cv2.VideoCapture("/home/chen/视频/ウェブカム/2021-06-19-221908.webm") 


fourcc = cv2.VideoWriter_fourcc(*'XVID')

out=cv2.VideoWriter("/home/chen/桌面/Bento.mp4",fourcc,20.0,(640,480))

ret, frame = camera.read()
while ret:
  # img1 = cv2.imread("/home/chen/tem_picture/Screenshot from 2021-05-27 02-28-49.png",1)


  # MIN_MATCH = 10
  # # ORB 检测器生成  ---①
  # detector = cv2.ORB_create(1000)
  # # Flann 创建提取器 ---②
  # FLANN_INDEX_LSH = 6
  # # dict() 函数用于创建一个字典。典也是Python语言中经常使用的一种数据类型。跟列表类似，字典是另外一种可存储任意类型的数据，并且字典储存的数据也是可以修改的
  # index_params= dict(algorithm = FLANN_INDEX_LSH,
  #                   table_number = 6,
  #                   key_size = 12,
  #                   multi_probe_level = 1)
  # search_params=dict(checks=32)

  # # FLANN特征匹配
  # matcher = cv2.FlannBasedMatcher(index_params, search_params)
  fgmask = mog.apply(frame)
  th = cv2.threshold(np.copy(fgmask), 244, 255, cv2.THRESH_BINARY)[1]
   
  th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 2)
  dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,3)), iterations = 5)
   
  image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for c in contours:
    if cv2.contourArea(c) > 1000:
      (x,y,w,h) = cv2.boundingRect(c)
      cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

  # img1 = cv2.imread("/home/chen/图片/Screenshot from 2021-05-28 15-54-48.png", cv2.IMREAD_GRAYSCALE)
  

  # img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # # ORB Detector
  # orb = cv2.ORB_create()
  # kp1, des1 = orb.detectAndCompute(img1, None)
  # kp2, des2 = orb.detectAndCompute(img2, None)

  # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  # matches = bf.match(des1, des2)
  # matches = sorted(matches, key = lambda x:x.distance)

  # matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
  # img2 = frame & cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
  # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  # # 得到特征点和特征点描述
  # # kp1 是一个包含若干点的列表
  # # desc1 对应每个点的描述符 是一个列表， 每一项都是检测到的特征的局部图像
  # kp1, desc1 = detector.detectAndCompute(gray1, None)
  # kp2, desc2 = detector.detectAndCompute(gray2, None)
  # # k=2 knnMatch 比率测试
  # # KNNMatch，可设置K = 2 ，即对每个匹配返回两个最近邻描述符，仅当第一个匹配与第二个匹配之间的距离足够小时，才认为这是一个匹配。
  # bf = cv2.BFMatcher()
  # matches = bf.knnMatch(desc1,desc2, k=2)
  # # matches = matcher.knnMatch(desc1, desc2, 2)#不是的2的话测不出来，暂时不知道原因
  # # 提取邻近点距离为75％的良好匹配点---②
  # ratio = 0.85
  # # 表达式或字符串最末尾是 \ 的话，一定代表 换行 继续写（即使是r也无法改变），必须要向下边空一行，程序才能跑
  # good_matches = [m[0] for m in matches \
  #                     if len(m) == 2 and m[0].distance < m[1].distance * ratio]
  # print(good_matches)
  # # print('good matches:%d/%d' %(len(good_matches),len(matches)))
  # # 모用0填充蒙版，以防止绘制所有匹配点
  # matchesMask = np.zeros(len(good_matches)).tolist()
  # # 如果匹配点数超过最小数量
  # if len(good_matches) > MIN_MATCH: 
  #     # 查找具有良好匹配点的源图像和目标图像的坐标 ---③
  #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]) #查询图像的特征描述子索引
  #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
  #     # 生成变换矩阵 ---⑤
  #     mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
  #     accuracy=float(mask.sum()) / mask.size
  #     print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))
  #     if mask.sum() > MIN_MATCH:  # 如果大于正常匹配点的最小数量
  #         # 遮罩设置仅绘制异常匹配点
  #         matchesMask = mask.ravel().tolist() #ravel()多维数组转换为一维数组   tolist()将矩阵(matrix)和数组(array)转化为列表
  #         # 透视转换为原始图像坐标后的显示区域  ---⑦
  #         h,w, = img1.shape[:2]
  #         pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
  #         dst = cv2.perspectiveTransform(pts,mtrx)
  #         img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
  # # 用遮罩绘制匹配点 ---⑨
  # res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
  #                     matchesMask=matchesMask,
  #                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

 
  cv2.imshow("mog", fgmask)
  cv2.imshow("thresh", th)
  cv2.imshow("diff", frame & cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))
  out.write(frame & cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))
  cv2.imshow("detection", frame)

  # cv2.imshow("matching_result", matching_result)
  ret, frame = camera.read()  #读取视频帧数据
  if cv2.waitKey(100) & 0xff == ord("q"):
      break
 
camera.release()
cv2.destroyAllWindows()
