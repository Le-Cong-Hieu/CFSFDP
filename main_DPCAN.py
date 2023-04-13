from distance.distance_builder import *
from distance.distance import *
import numpy as np
import cv2
import os
import logging
from plot import *
from cluster import *
from sklearn.metrics import confusion_matrix

def distance_crack(path_img,):
  builder = DistanceBuilder()
  #builder.load_points(r'data/spiral.txt')
  img =cv2.imread(path_img)
  if len(img.shape)>2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  pointcrack=np.argwhere(img>0)
  
  builder.load_point_crack(pointcrack)
  builder.build_distance_file_for_cluster(SqrtDistance(), r'data/spiral_distance.dat')
  return img

def cluster_crack(data, density_threshold, distance_threshold,img, auto_select_dc = False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dpcluster = DensityPeakCluster()
    rho, delta, nneigh = dpcluster.cluster(load_paperdata, data, density_threshold, distance_threshold, auto_select_dc = auto_select_dc)
    #logger.info(str(len(dpcluster.ccenter)) + ' center as below')
    # for idx, center in dpcluster.ccenter.items():
    #     logger.info('%d %f %f' %(idx, rho[center], delta[center]))
    # plot_rho_delta(rho, delta)   #plot to choose the threthold
    # plot_rhodelta_rho(rho,delta)
    dp_mds,cls = plot_cluster(dpcluster)
    x,y=dp_mds[:, 0], dp_mds[:, 1]
    pointcrack=np.argwhere(img>0)
    if cls is None:
        plt.scatter(x,y)
		#plt.plot(x, y, color=styles[0], linestyle='-', marker='.')
    else:
        clses = set(cls)
        xs, ys = {}, {}
        for i in range(len(x)):
            try:
                xs[cls[i]].append(pointcrack[i,0])
                ys[cls[i]].append(pointcrack[i,1])
            except KeyError:
                xs[cls[i]] = [pointcrack[i,0]]
                ys[cls[i]] = [pointcrack[i,1]]
        added = 1
        #print(x.shape,cls.shape)
        colors = np.random.rand(100)
        k=0
        
        img_out=np.zeros(img.shape)
        for idx, cls in enumerate(clses):
            if cls == -1:
                added = 0
                #plt.scatter(xs[cls], ys[cls], c='k',cmap='cool',s=1)
            else:
                #colors = np.random.rand(100)
                #print((ys[cls], xs[cls]))
                cv2.circle(img_out,(ys[cls][0], xs[cls][0]),1,(0,255,0),-1)
                #cv2.circle(img,(y[k], x[k]))
                style = colors[idx + added]
                #plt.scatter(xs[cls], ys[cls], c=style,cmap='cool')
            k=k+1
			
			#plt.plot(xs[cls], ys[cls], color=style, linestyle='None', marker='.')
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.show()
    return img_out
def F1_rc_pr(seg,gt):
    #print(seg.shape,gt.shape)
    if len(seg.shape)>2:
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    if len(gt.shape)>2:
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    gt_cp=np.zeros(gt.shape)
    gt_cp[gt==76]=255
    # cv2.imshow('gt',gt_cp)
    # cv2.waitKey(0)
    seg[seg>0]=255
    # cv2.imshow('seg',seg)
    # cv2.waitKey(0)
    f1,rc,pr=Accuracy1(gt_cp,seg)
    return f1,rc,pr 

def Accuracy1(GT,seg):
    r = []
    p = []
    F = 0
    #[x,y] = np.argwhere(GT >0)
    GT[np.argwhere(GT >0)] = 1
    GT = np.ndarray.flatten(GT)
    #[x,y] = np.argwhere(seg > 0)
    seg[np.argwhere(seg > 0)] = 1
    seg = np.ndarray.flatten(seg)
    CM = confusion_matrix(GT,seg)
    c = np.shape(CM)
    for i in range(c[1]):
        if (np.sum(CM[i,:]) == 0):
            r.append(0)
        else:
            a = CM[i,i]/(np.sum(CM[i,:]))
            r.append(a)
        if (np.sum(CM[:,i]) == 0):
            p.append(0)
        else:
            p.append(CM[i,i]/(np.sum(CM[:,i])))
    F = 2*(np.mean(r)*np.mean(p))/(np.mean(p)+np.mean(r))
    return F,np.mean(p),np.mean(r)


if __name__ == '__main__':
    path_img='/Users/minhhieu/vs_code/DensityPeakCluster/data/gt/0003-3.png'
    name_img=(os.path.split(path_img)[-1]).split(".")[0]
    path_gt='data/seg/'+name_img+'.png'


    gt=cv2.imread(path_gt)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

    img=distance_crack('/Users/minhhieu/vs_code/DensityPeakCluster/data/gt/0003-3.png')

    f1_max=0
    density=0
    distance =0
    for density_threshold in range(1,20,1):
        for distance_threshold in range(1,20,1):
            print(density_threshold,distance_threshold/10)
            img_out=cluster_crack('data/spiral_distance.dat', density_threshold, distance_threshold/10,img,True)
            f1,rc,p=F1_rc_pr(img_out,gt)
            print(f1,rc,p)
            if f1>f1_max:
                f1_max=f1
                print(f1_max)
                distance=distance_threshold
                density=density_threshold
    print(distance,density)
    print(f1)


            
#   builder = DistanceBuilder()
#   #builder.load_points(r'data/spiral.txt')
#   img =cv2.imread('/Users/minhhieu/vs_code/DensityPeakCluster/data/gt/0003-2.png')
#   if len(img.shape)>2:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   pointcrack=np.argwhere(img>0)
#   print(pointcrack)
#   builder.load_point_crack(pointcrack)
#   builder.build_distance_file_for_cluster(SqrtDistance(), r'data/spiral_distance.dat')