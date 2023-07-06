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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>hm:array<L>;@group(0)@binding(3)var<storage,read_write>ao:eC;@group(0)@binding(4)var<storage,read_write>bM:array<cJ>;@group(0)@binding(5)var<storage,read_write>X:array<bt>;const o=256u;var<workgroup>be:array<j,o>;var<workgroup>jh:j;var<workgroup>fC:j;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fC=atomicLoad(&ao.aq);}let aq=workgroupUniformLoad(&fC);if(aq&eB)!=e{return;}let aC=d/h(cQ);let bs=d/h(bu);let aG=N.x;var bo=cM;if aG<n.cN{bo=u[n.c5+aG];}var J=0;var S=0;var T=0;var V=0;if bo!=cM&&bo!=es{let l=hm[aG];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bs));T=m(ceil(l.z*aC));V=m(ceil(l.w*bs));}}let fH=j(clamp(J,0,m(n.aK)));let fG=j(clamp(S,0,m(n.cO)));let fF=j(clamp(T,0,m(n.aK)));let fE=j(clamp(V,0,m(n.cO)));let b8=(fF-fH)*(fE-fG);var d6=b8;be[k.x]=b8;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){d6+=be[k.x-(f<<i)];}workgroupBarrier();be[k.x]=d6;}if k.x==o-f{let b9=be[o- f];var a5=atomicAdd(&ao.F,b9);if a5+b9>n.iz{a5=e;atomicOr(&ao.aq,fw);}bM[aG].X=a5;}storageBarrier();let fD=bM[aG|(o- f)].X;storageBarrier();if aG<n.cN{let ho=select(e,be[k.x- f],k.x>e);let l=vec4(fH,fG,fF,fE);let W=cJ(l,fD+ho);bM[aG]=W;}let hn=be[o- f];for(var i=k.x;i<hn;i+=o){X[fD+i]=bt(0,e);}}`
