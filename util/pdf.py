# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:59:57 2020
@author: KuanChao Chu

Topic:
    https://www.blog.pythonlibrary.org/2018/02/06/reportlab-101-the-textobject/
"""
import time
from PIL import Image as PILImage
from pathlib import Path
#from reportlab.pdfbase import pdfmetrics 
#from reportlab.lib.styles import getSampleStyleSheet
#from reportlab.platypus import SimpleDocTemplate, Paragraph, Image


def pdf_gen(f_path, text_list, figure_list=None, figure_height=200):
    """ 
    text_list is the list of each line
    
    (got image compression...)
    """
    story = [] 
    style_sheet = getSampleStyleSheet()
    normal_style = style_sheet['Normal']
    
    for line in text_list:
        line_re = line.replace(' ', '&nbsp;')
        story.append(Paragraph(line_re, normal_style))
        
    def get_img_w_h_ratio(image_path):
        img = PILImage.open(image_path)
        w, h = img.size
        return w / h    
    
    if figure_list:
        for figure in figure_list:
            ratio = get_img_w_h_ratio(figure)
            story.append(Image(figure, width=figure_height*ratio, height=figure_height))
    
    fname = Path(f_path).stem
    suffix = time.strftime("%Y%m%d%H%M", time.localtime())[2:]
    out_fname = '{}_{}.pdf'.format(fname, suffix)
    
    doc = SimpleDocTemplate(str(Path(f_path).parent / out_fname))    
    doc.build(story)
    

if __name__ == '__main__':

    print('pdf.py')
    #pdf_gen(str(Path.cwd()), ['sssssss\n']+['1\n']*40, ['loss.jpg']) 
    