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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>hl:array<L>;@group(0)@binding(3)var<storage,read_write>ao:eE;@group(0)@binding(4)var<storage,read_write>bM:array<cJ>;@group(0)@binding(5)var<storage,read_write>W:array<bs>;const o=256u;var<workgroup>be:array<j,o>;var<workgroup>jh:j;var<workgroup>fE:j;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fE=atomicLoad(&ao.aq);}let aq=workgroupUniformLoad(&fE);if(aq&eD)!=e{return;}let aC=d/h(cQ);let bq=d/h(bu);let aG=N.x;var bn=cM;if aG<n.cN{bn=u[n.c5+aG];}var J=0;var S=0;var T=0;var V=0;if bn!=cM&&bn!=eu{let l=hl[aG];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bq));T=m(ceil(l.z*aC));V=m(ceil(l.w*bq));}}let fJ=j(clamp(J,0,m(n.aK)));let fI=j(clamp(S,0,m(n.cO)));let fH=j(clamp(T,0,m(n.aK)));let fG=j(clamp(V,0,m(n.cO)));let ca=(fH-fJ)*(fG-fI);var d8=ca;be[k.x]=ca;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){d8+=be[k.x-(f<<i)];}workgroupBarrier();be[k.x]=d8;}if k.x==o-f{let cb=be[o- f];var a5=atomicAdd(&ao.F,cb);if a5+cb>n.iz{a5=e;atomicOr(&ao.aq,fy);}bM[aG].W=a5;}storageBarrier();let fF=bM[aG|(o- f)].W;storageBarrier();if aG<n.cN{let hn=select(e,be[k.x- f],k.x>e);let l=vec4(fJ,fI,fH,fG);let X=cJ(l,fF+hn);bM[aG]=X;}let hm=be[o- f];for(var i=k.x;i<hm;i+=o){W[fF+i]=bs(0,e);}}`
