import numpy as np
import cv2
import uuid

vdo_path = 'D:/Projectz/vdeo/test/walk/'
vdo_dir_name = 'newvdo'
vdo_dir_name = str(input("Enter VDO_DIR_NAME <= "))
vdo_path += vdo_dir_name + '/'
unique_filename = str(uuid.uuid4())


depth_stream = cv2.VideoCapture(vdo_path + "depth.avi")
ir_stream = cv2.VideoCapture(vdo_path + "ir.avi")

# find x,y,w,h of each frame
class cvRect:
    def __init__(self, xywh):
        self.x = xywh[0]
        self.y = xywh[1]
        self.w = xywh[2]
        self.h = xywh[3]
        self.xmin = self.x
        self.ymin = self.y
        self.xmax = self.x + self.w
        self.ymax = self.y + self.h
    def area(self):
        return self.w * self.h
    def tl(self):
        return [self.x,self.y]
    def br(self):
        return [self.x+self.w,self.y+self.h]
    def center(self):
        return [self.x+(self.w/2),self.y+(self.h/2)]
    def get_xywh(self):
        return  [self.x,self.y,self.w,self.h]

dictLabel = {
    'person':0,

}

def makeLabelYOLO(xywh,nameClass,IMAGE_SIZE):
    ''' makeLabelYOLO(xywh<-cvRect,nameClass<-string,IMAGE_SIZE<-numpy.shape())
    '''
    # Yolo Format
    ''' Label_ID X_CENTER_NORM Y_CENTER_NORM WIDTH_NORM HEIGHT_NORM Label_ID_2 X_CENTER_NORM
    X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH 
    Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT 
    WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH 
    HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT
    '''
    global cvRect,dictLabel
    IMAGE_WIDTH = IMAGE_SIZE[1]
    IMAGE_HEIGHT = IMAGE_SIZE[0]
    # find Label_ID
    Label_ID = dictLabel[nameClass]
    X_CENTER_NORM = xywh.center()[0]/IMAGE_WIDTH 
    Y_CENTER_NORM = xywh.center()[1]/IMAGE_HEIGHT 
    WIDTH_NORM = xywh.w/IMAGE_WIDTH 
    HEIGHT_NORM = xywh.h/IMAGE_HEIGHT
    return "%d %.6f %.6f %.6f %.6f" % (Label_ID,X_CENTER_NORM,Y_CENTER_NORM,WIDTH_NORM,HEIGHT_NORM)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=None, varThreshold=None, detectShadows=True)
# fgbg.setHistory(600)
# fgbg.setNMixtures(5)
fgbg.setDetectShadows(False)
fgbg.setVarThreshold(95)

while True:
    ret, frame_depth = depth_stream.read()
    ret, frame_ir = ir_stream.read()

    # Decode Depth
    # 8upperbits data in ch1 / 8lowerbits data in ch2 / ignore ch3
    depth_ch1, depth_ch2, _ = cv2.split(frame_depth)
    # 8upperbits data -> convert uint16 and shift left << 8
    decoded_depth = np.left_shift(np.uint16(depth_ch1.copy()), 8)
    decoded_depth = np.bitwise_or(decoded_depth, np.uint16(
        depth_ch2.copy()))

    decoded_depth_8bits = np.uint8(decoded_depth*10 / 257)
    decoded_depth_8bits_imgsize = decoded_depth_8bits.shape[:2]
    

    # Decode IR
    # 8upperbits data in ch1 / 8lowerbits data in ch2 / ignore ch3
    ir_ch1, ir_ch2, _ = cv2.split(frame_ir)
    # 8upperbits data -> convert uint16 and shift left << 8
    decoded_ir = np.left_shift(np.uint16(ir_ch1.copy()), 8)
    decoded_ir = np.bitwise_or(decoded_ir, np.uint16(
        ir_ch2.copy()))  # bitwise or with 8lowerbits

    decoded_ir_8bits = np.uint8(decoded_ir*400 / 257)
    decoded_ir_8bits_imgsize = decoded_ir_8bits.shape[:2]

    # Adjusted for display
    decoded_depth *= 10
    decoded_ir *= 400

    # Get frame id
    frameID = depth_stream.get(cv2.CAP_PROP_POS_FRAMES)

    # Learning background
    if frameID <= 10:
        fgmask = fgbg.apply(decoded_depth, learningRate=0.8)

    elif frameID <= 100:
        fgmask = fgbg.apply(decoded_depth, learningRate=0.5)

    else:
        fgmask = fgbg.apply(decoded_depth, learningRate=0)

    # Erode
    rect3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cross3x3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # ellipse3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    rect5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # cross5x5 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    # ellipse5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    rect9x9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # Mop
    # anti_noise = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, rect5x5, iterations=1)

    # eroded_rect5x5_img = cv2.erode(fgmask, rect5x5, iterations = 1)
    # eroded_cross5x5_img = cv.erode(bin_img, cross5x5, iterations = 1)
    # eroded_ellipse5x5_img = cv.erode(bin_img, ellipse5x5, iterations = 1)
    anti_noise = cv2.erode(fgmask, rect9x9, iterations=1)

    # Dilate
    anti_noise = cv2.dilate(anti_noise, rect9x9, iterations=1)

    # Contour
    # canny = cv2.Canny(anti_noise, 30, 100)
    lapas = cv2.Laplacian(anti_noise, cv2.CV_16S, ksize=3)
    lapas = np.uint8(lapas / 257)
    # print(lapas.dtype)
    contours, hierarchy = cv2.findContours(
        lapas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn_contour = np.zeros(lapas.shape, dtype=np.uint8)
    drawn_contour = cv2.cvtColor(drawn_contour, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(drawn_contour, contours, -1, (0, 255, 200), 2)

    max_area = 0
    max_contourID = -1
    # Find bounary
    for index, val in enumerate(contours):
        x, y, w, h = cv2.boundingRect(val)
        area = w * h

        if area > max_area and area > 10000:
            max_area = area
            max_contourID = index

    if max_contourID != -1:
        x, y, w, h = cv2.boundingRect(contours[max_contourID])
        cv2.rectangle(anti_noise, (x, y), (x+w, y+h),
                      (255, 255, 255), thickness=2)
        # print(f"w*h = {w*h}")
        debug_contour = np.zeros(lapas.shape, dtype=np.uint8)
        debug_contour = cv2.cvtColor(debug_contour, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(debug_contour, contours,
                         max_contourID, (0, 255, 200), 2)

        # Fill
        bg_fill = np.zeros(fgmask.shape, dtype=np.uint8)
        bg_fill = cv2.fillPoly(
            bg_fill, pts=[contours[max_contourID]], color=(255, 255, 255))

        # Crop
        anti_noise_crop = bg_fill[y:y+h, x:x+w]
        deDepth_person = decoded_depth[y:y+h, x:x+w]
        _, personBin = cv2.threshold(
            anti_noise_crop, 200, 255, cv2.THRESH_BINARY)
        personBin_16bits = np.uint16(personBin) * 257
        person = np.bitwise_and(
            personBin_16bits, deDepth_person)  # 16bits = 65_535
        person_8bits = np.uint8(person / 257)
        deDepth_person8bits = np.uint8(deDepth_person / 257)

        # About IR
        deIR_person = decoded_ir[y:y+h, x:x+w]
        IRperson = np.bitwise_and(personBin_16bits, deIR_person)
        IRperson_8bits = np.uint8(IRperson / 257)
        deIR_person8bits = np.uint8(deIR_person / 257)

        # BgFill
        # bg_fill_crop = bg_fill[y:y+h, x:x+w]
        # BGFill_deDepth_person = decoded_depth[y:y+h, x:x+w]
        # _, BgFill_personBin = cv2.threshold(bg_fill_crop, 200, 255, cv2.THRESH_BINARY)
        # BgFill_personBin_16bits = np.uint16(BgFill_personBin) * 257
        # BgFill_person = np.bitwise_and(BgFill_personBin_16bits, BGFill_deDepth_person) # 16bits = 65_535
        # BgFill_person_8bits = np.uint8(BgFill_person / 257)
        # deDepth_person8bits = np.uint8(deDepth_person / 257)

        # Im write --------------------------------------------------------
        # Depth

        # cv2.imwrite('D:/Projectz/label/datatest/Depth_Bg/' + vdo_dir_name +
        #             str(int(frameID)) + '.png', decoded_depth_8bits)   # Write images
        # cv2.imwrite('./image/Depth_Bg/test/walk/' + vdo_dir_name +
        #             str(int(frameID)) + '.png', deDepth_person8bits)  # With BG
        # save img-depth
        cv2.imwrite('D:/Projectz/label_test/bg/' + unique_filename + str(int(frameID)) + '.png', decoded_depth_8bits)
        # save img-ir
        cv2.imwrite('D:/Projectz/label_test/ir/' + unique_filename + str(int(frameID)) + '.png',  decoded_ir_8bits_imgsize)    
        # IR
        # cv2.imwrite('./image/Ir_NoBg/test/walk/' + vdo_dir_name +
        #             str(int(frameID)) + '.png', IRperson_8bits)
        # cv2.imwrite('./image/Ir_Bg/test/walk/' + vdo_dir_name +
        #             str(int(frameID)) + '.png', deIR_person8bits)


        # yolo_label = makeLabelYOLO(cvRect([x,y,w,h]),'person',decoded_depth_8bits_imgsize)
        # text_file = open('D:/Projectz/label/datatest/Depth_Bg/' + vdo_dir_name +
        #             str(int(frameID)) + '.txt', "w")
        # n = text_file.write(yolo_label)
        # text_file.close()

        # save text-depth
        yolo_label = makeLabelYOLO(cvRect([x,y,w,h]),'person',decoded_depth_8bits_imgsize)
        text_file = open('D:/Projectz/label_test/bg/' + unique_filename +
                    str(int(frameID)) + '.txt', "w")
        n = text_file.write(yolo_label)
        text_file.close()

         # save text-ir
        yolo_label = makeLabelYOLO(cvRect([x,y,w,h]),'person',decoded_ir_8bits_imgsize)
        text_file = open('D:/Projectz/label_test/ir/' + unique_filename +
                    str(int(frameID)) + '.txt', "w")
        n = text_file.write(yolo_label)
        text_file.close()
        

        # Imshow ------------------------------------------------------------------------------------------
        # cv2.imshow('Person', person)
        cv2.imshow('Depth Person', deDepth_person8bits)
        # cv2.imshow('Anti-noise Crop', anti_noise_crop)
        # cv2.imshow('Debug_contour', debug_contour)
        # cv2.imshow('BgFill', bg_fill)
        cv2.imshow('Person_8bits', person_8bits)
        cv2.imshow('IRperson_8bits', IRperson_8bits)
        # cv2.imshow('BgFill_8bits', BgFill_person_8bits)

    cv2.imshow('Original', fgmask)
    # cv2.imshow('Anti-noise', anti_noise)
    # cv2.imshow('Lapas', lapas)
    # cv2.imshow('drawn_contour', drawn_contour)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # if frameID == 124:
    #     cv2.waitKey()

depth_stream.release()
ir_stream.release
cv2.destroyAllWindows()
