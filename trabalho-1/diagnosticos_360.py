from pathlib import Path
import json,csv,sys,hashlib
import numpy as np,cv2 as cv
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser(description='Diagnósticos do panorama 360 gerados a partir dos resultados salvos.')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--input',type=Path,required=True)
args=parser.parse_args()
p=args.output;d=json.loads((p/'cameras_ajustadas.json').read_text());raw=json.loads((p/'correspondencias.json').read_text());rf=json.loads((p/'correspondencias_rotacao.json').read_text());meta=json.loads((p/'render.json').read_text());seams=np.load(p/'seams.npz')
def csvwrite(name,rows):
 with (p/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
cs=d['cameras'];order=sorted(cs,key=lambda c:c['yaw']);csvwrite('ordem_circular.csv',[dict(posicao=i+1,imagem=c['file'],yaw_graus=c['yaw'],focal_px=c['focal']) for i,c in enumerate(order)])
csvwrite('keypoints.csv',[dict(imagem=c['file'],keypoints=c['keypoints'],sha256=c['sha256']) for c in cs]);csvwrite('pares_ajustados.csv',d['edges']);csvwrite('pares_todos.csv',[{k:e[k] for k in ('source','target','inliers','matches','confidence')} for e in raw['pairs']]);csvwrite('contribuicao.csv',[dict(imagem=c['file'],pixels_mascara_costura=int(np.count_nonzero(seams[str(i)]))) for i,c in enumerate(cs)])
fig,ax=plt.subplots(figsize=(12,4));ax.bar(range(1,23),[c['keypoints'] for c in order]);ax.set(xlabel='Vista na ordem circular estimada',ylabel='Keypoints SIFT',title='Panorama 360° — detecção por imagem');fig.tight_layout();fig.savefig(p/'keypoints.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,8));positions={c['file']:np.array([np.cos(2*np.pi*i/len(cs)),np.sin(2*np.pi*i/len(cs))]) for i,c in enumerate(order)}
for e in d['edges']:
 a,b=positions[e['source']],positions[e['target']];ax.plot([a[0],b[0]],[a[1],b[1]],color='#7aabc4',alpha=.5,lw=1)
for i,c in enumerate(order):
 x,y=positions[c['file']];ax.scatter(x,y,color='#214e65',s=40);ax.text(x*1.13,y*1.13,str(i+1),ha='center',va='center')
ax.set_aspect('equal');ax.axis('off');ax.set_title('Grafo após ajuste: 22 vistas, '+str(len(d['edges']))+' pares\nOrdem angular estimada; rótulos em ordem_circular.csv');fig.tight_layout();fig.savefig(p/'grafo_circular.png',dpi=150);plt.close(fig)
errs=[e['median_ray_error_px'] for e in d['edges']];fig,ax=plt.subplots(figsize=(10,3));ax.bar(range(1,len(errs)+1),errs);ax.set(xlabel='Par aceito',ylabel='Mediana do resíduo angular × focal (px)',title='Ajuste global robusto — diagnóstico por par');fig.tight_layout();fig.savefig(p/'erros_ajuste.png',dpi=160);plt.close(fig)
yaws={c['file']:c['yaw'] for c in cs};closure=min(d['edges'],key=lambda e:abs((yaws[e['source']]-yaws[e['target']]+180)%360-180));mask=cv.imread(str(p/'mascara.png'),0);image=cv.imread(str(p/'panorama_360.jpg'));assert image.shape[:2]==mask.shape and np.all(mask>0)
verified=all(hashlib.sha256((args.input/c['file']).read_bytes()).hexdigest()==c['sha256'] for c in cs);assert verified
summary=dict(location='Bragança Paulista (SP)',camera='Samsung Galaxy S24 Ultra',images=22,contributing_images=sum(np.any(seams[str(i)]) for i in range(22)).item(),accepted_pairs=len(d['edges']),rotation_ransac_rejected_pairs=sum(e['confidence']>1 and e.get('homography_inliers',0)>=25 and e['inliers']<25 for e in rf['pairs']),closure_pair=closure,optimizer=d['optimizer'],full_longitude_coverage=bool(np.all(mask>0)),horizontal_degrees=360,vertical_degrees=meta['vertical_degrees'],width=meta['width'],height=meta['height'],source_hashes_verified=verified,manual_correspondences=0,manual_source_regions=0,limitations=['Paralaxe residual em objetos próximos','Desalinhamentos finos em fios e calçada','Movimento de carros entre vistas','Cobertura vertical parcial, sem zênite ou nadir'])
(p/'verificacao.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
