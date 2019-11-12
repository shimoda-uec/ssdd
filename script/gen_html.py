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
fn='../data/trainaug_id.txt'
f = open(fn,'r')
image_ids = f.read().splitlines()
f.close()
print('$(function(){')
print('var arr=[')
for i in range(len(image_ids)):
    print('"{}",'.format(image_ids[i]))
print('];')
print('var num=0;')
shown=5
btn_num=[-1*shown*10, -shown,-1,"+"+"1","+"+str(shown),"+"+str(shown*10)]
btn_if=['if (num>=0){','if (num>=0){','if (num>=0){','if (num<10582){','if (num<10582){','if (num<10582){']
for b in range(6):
    print('$(\'.btn{}\').on(\'click\', function()'.format(b)+"{")
    print(btn_if[b])
    print('num = num{};'.format(btn_num[b]))
    print('}')
    for i in range(shown):
        print('$(".image{}").children("img").attr("src", "../voc_root/JPEGImages/"+arr[{}+num]+".jpg");'.format(i,i))
        print('$(".aff{}").children("img").attr("src", "../prepare_labels/results/out_aff/"+arr[{}+num]+".png");'.format(i,i))
        print('$(".crf{}").children("img").attr("src", "../prepare_labels/results/out_aff_crf/"+arr[{}+num]+".png");'.format(i,i))
    print('});')
print('})')
print('</script>')
btn_txt=['<<<','<<','<','>','>>','>>>']
for i in range(6):
    print('<div class="btn{}" style="display: inline">'.format(i))
    print('<button >'+btn_txt[i]+'</button>')
    print('</div>')
print('<table align="center" style="font-size : 10px;" border="0" cellspacing="0" >')
w=200
h=200
print('<tr><td>image</td><td>PSA</td><td>PSA with CRF</td></tr>')
for i in range(shown):
    print('<tr>')
    print('<td>')
    print('<div class="image{}">'.format(i))
    print('<img src="../voc_root/JPEGImages/{}.jpg" width="{}" height="{}">'.format(image_ids[i], w,h))
    print('</div>')
    print('</td>')
    print('<td>')
    print('<div class="aff{}">'.format(i))
    print('<img src="../prepare_labels/results/out_aff/{}.png" width="{}" height="{}">'.format(image_ids[i], w,h))
    print('</div>')
    print('</td>')
    print('<td>')
    print('<div class="crf{}">'.format(i))
    print('<img src="../prepare_labels/results/out_aff_crf/{}.png" width="{}" height="{}">'.format(image_ids[i], w,h))
    print('</div>')
    print('</td>')
    print('</tr>')
print('</body></html>')
