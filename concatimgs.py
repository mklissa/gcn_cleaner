from PIL import Image as im
import numpy as np
import pdb
# names = ['800','nowall','nowallfeatures','wall']
# appendix = ['','_rounded']

names = ['fourroom_']
appendix= ['169','484','1276','1320']
for name in names:

	imgs=[]
	for app in appendix:
		imgs.append(np.array(im.open("{}{}_results.png".format(name,app))))
	
	# end= np.hstack(imgs)
	end0= np.hstack(imgs[:2])
	end1= np.hstack(imgs[2:])
	end= np.vstack([end0,end1])
	imgs_comb = im.fromarray( end)
	imgs_comb.save( '{}concat.png'.format(name) )   



# appendix= ['0','5','15','30','150','180']
# imgs=[]
# for app in appendix:
# 	imgs.append(np.array(im.open("afteroption/{}.png".format(app))))

# end= np.hstack(imgs)
# imgs_comb = im.fromarray( end)
# imgs_comb.save( 'vfdiff.png' )   

