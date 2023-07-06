/* eslint-disable */
import tile from './shared/tile.js';
import drawtag from './shared/drawtag.js';
import bump from './shared/bump.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${bump}
${drawtag}
${tile}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>hl:array<L>;@group(0)@binding(3)var<storage,read_write>ao:eD;@group(0)@binding(4)var<storage,read_write>bL:array<cJ>;@group(0)@binding(5)var<storage,read_write>X:array<bs>;const o=256u;var<workgroup>be:array<j,o>;var<workgroup>jh:j;var<workgroup>fD:j;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fD=atomicLoad(&ao.aq);}let aq=workgroupUniformLoad(&fD);if(aq&eC)!=e{return;}let aC=d/h(cQ);let bq=d/h(bt);let aG=N.x;var bn=cM;if aG<n.cN{bn=u[n.c5+aG];}var J=0;var S=0;var T=0;var V=0;if bn!=cM&&bn!=et{let l=hl[aG];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bq));T=m(ceil(l.z*aC));V=m(ceil(l.w*bq));}}let fI=j(clamp(J,0,m(n.aK)));let fH=j(clamp(S,0,m(n.cO)));let fG=j(clamp(T,0,m(n.aK)));let fF=j(clamp(V,0,m(n.cO)));let b8=(fG-fI)*(fF-fH);var d7=b8;be[k.x]=b8;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){d7+=be[k.x-(f<<i)];}workgroupBarrier();be[k.x]=d7;}if k.x==o-f{let b9=be[o- f];var a5=atomicAdd(&ao.F,b9);if a5+b9>n.iz{a5=e;atomicOr(&ao.aq,fx);}bL[aG].X=a5;}storageBarrier();let fE=bL[aG|(o- f)].X;storageBarrier();if aG<n.cN{let hn=select(e,be[k.x- f],k.x>e);let l=vec4(fI,fH,fG,fF);let W=cJ(l,fE+hn);bL[aG]=W;}let hm=be[o- f];for(var i=k.x;i<hm;i+=o){X[fE+i]=bs(0,e);}}`
