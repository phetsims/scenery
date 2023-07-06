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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>hn:array<L>;@group(0)@binding(3)var<storage,read_write>ao:eD;@group(0)@binding(4)var<storage,read_write>bM:array<cJ>;@group(0)@binding(5)var<storage,read_write>X:array<bt>;const o=256u;var<workgroup>be:array<j,o>;var<workgroup>ji:j;var<workgroup>fE:j;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fE=atomicLoad(&ao.aq);}let aq=workgroupUniformLoad(&fE);if(aq&eC)!=e{return;}let aC=d/h(cQ);let bs=d/h(bu);let aG=N.x;var bo=cM;if aG<n.cN{bo=u[n.c5+aG];}var J=0;var S=0;var T=0;var V=0;if bo!=cM&&bo!=et{let l=hn[aG];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bs));T=m(ceil(l.z*aC));V=m(ceil(l.w*bs));}}let fJ=j(clamp(J,0,m(n.aK)));let fI=j(clamp(S,0,m(n.cO)));let fH=j(clamp(T,0,m(n.aK)));let fG=j(clamp(V,0,m(n.cO)));let b9=(fH-fJ)*(fG-fI);var d7=b9;be[k.x]=b9;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){d7+=be[k.x-(f<<i)];}workgroupBarrier();be[k.x]=d7;}if k.x==o-f{let ca=be[o- f];var a5=atomicAdd(&ao.F,ca);if a5+ca>n.iA{a5=e;atomicOr(&ao.aq,fy);}bM[aG].X=a5;}storageBarrier();let fF=bM[aG|(o- f)].X;storageBarrier();if aG<n.cN{let hp=select(e,be[k.x- f],k.x>e);let l=vec4(fJ,fI,fH,fG);let W=cJ(l,fF+hp);bM[aG]=W;}let ho=be[o- f];for(var i=k.x;i<ho;i+=o){X[fF+i]=bt(0,e);}}`
