import os
print('<!DOCTYPE html>')
print('<html>')
print('<head>')
print('<meta http-equiv="Content-Type" content="text/html" charset="utf-8">')
print('<title>title</title>')
print('</head>')
print('<body>')
print('<script src="//code.jquery.com/jquery-1.11.1.min.js"></script>')
print('<script>')
fn='../voc_root/ImageSets/Segmentation/val.txt'
f = open(fn,'r')
image_ids = f.read().splitlines()
f.close()
modelid='default'
print('$(function(){')
print('var arr=[')
for i in range(len(image_ids)):
    print('"{}",'.format(image_ids[i]))
print('];')
print('var num=0;')
shown=5
btn_num=[-1*shown*10, -shown,-1,"+"+"1","+"+str(shown),"+"+str(shown*10)]
btn_if=['if (num>=0){','if (num>=0){','if (num>=0){','if (num<1500){','if (num<1500){','if (num<1500){']
for b in range(6):
    print('$(\'.btn{}\').on(\'click\', function()'.format(b)+"{")
    print(btn_if[b])
    print('num = num{};'.format(btn_num[b]))
    print('}')
    for i in range(shown):
        print('var n = num +1 + {};'.format(i))
        print('$(".image{}").children("img").attr("src", "../voc_root/JPEGImages/"+arr[{}+num]+".jpg");'.format(i,i))
        print('$(".seg{}").children("img").attr("src", "../validation/{}/seg_val_{}_"+n+".png");'.format(i,modelid,modelid,i))
        print('$(".crf{}").children("img").attr("src", "../validation/{}/seg_crf_val_{}_"+arr[{}+n]+".png");'.format(i,modelid,modelid,i))
        print('$(".gt{}").children("img").attr("src", "../voc_root/SegmentationClass/"+arr[{}+num]+".png");'.format(i,i))
    print('});')
print('})')
print('</script>')
btn_txt=['<<<','<<','<','>','>>','>>>']
for i in range(6):
    print('<div class="btn{}" style="display: inline">'.format(i))
    print('<button >'+btn_txt[i]+'</button>')
    print('</div>')
print('<table align="center" style="font-size : 11pt;" border="0" cellspacing="0" >')
w=200
h=200
print('<tr><td>Image</td><td>Inference</td><td>Ground truth</td></tr>')
for i in range(shown):
    print('<tr>')
    print('<td>')
    print('<div class="image{}">'.format(i))
    print('<img src="../voc_root/JPEGImages/{}.jpg" width="{}" height="{}">'.format(image_ids[i], w,h))
    print('</div>')
    print('</td>')
    print('<td>')
    print('<div class="seg{}">'.format(i))
    print('<img src="../validation/{}/seg_val_{}_{}.png" width="{}" height="{}");'.format(modelid,modelid,i+1, w,h))
    print('</div>')
    print('</td>')
    print('<td>')
    print('<div class="gt{}">'.format(i))
    print('<img src="../voc_root/SegmentationClass/{}.png" width="{}" height="{}">'.format(image_ids[i], w,h))
    print('</div>')
    print('</td>')
    print('</tr>')
print('</body></html>')
