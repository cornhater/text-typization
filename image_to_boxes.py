import os
import pytesseract
from PIL import Image
import cv2
import os

def image_to_boxes(img, out_name, outdir):
    # Get bounding box estimates
    boxes = pytesseract.image_to_data(img, lang='rus')
    y0 = 10**5
    y_max = 0
    h_max = 0
    boxes = boxes.splitlines()
    del boxes[0]
    flag = False
    count = 1
    for b in boxes:
        b = b.split()
        if int(b[5])==1:
            x0 = int(b[6])
        if int(b[5])!=0:
            flag = True
            x,y,w,h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            if y0 > y:
                y0 = y
            if h_max < h:
                h_max = h
            if y_max < y:
                y_max = y
        else:
            if flag == False:
                continue
            else:
                bbox = (x0, y0, x+w, y+h_max)
                working_slice = img.crop(bbox)
                print(working_slice.width)
                if (working_slice.height<50 
                    and working_slice.height>30 
                    and working_slice.width < 1500 
                    and working_slice.width > 500):
                    working_slice.save(os.path.join(outdir, out_name + "_slice_" + str(count).rjust(4, '0') + ".jpg"))
                y0 = 10**5
                flag = False
                y_max = 0
                h_max = 0
                count += 1
    
path_npa = (r'E:\botay\диплом\train_sample\npa')
strings_save_path = (r'E:\botay\диплом\genered_data')
    
def collect_strings(docs_path, strings_save_path, stop):
    count = 0
    images = os.listdir(docs_path)
    for doc in images:
        image_path = os.path.join(docs_path, doc)
        img = Image.open(image_path)
        image_to_boxes(img, 'doc_' + str(count).rjust(4, '0'), strings_save_path)
        count += 1
        saved = os.listdir(strings_save_path)
        if (len(saved) > stop): 
            for trash in saved[stop:]:
                os.remove(os.path.join(strings_save_path, trash))
            break
    
collect_strings(path_npa, strings_save_path, 100)