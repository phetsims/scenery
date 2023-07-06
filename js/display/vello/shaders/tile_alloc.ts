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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>s:aX;@group(0)@binding(2)var<storage>hl:array<L>;@group(0)@binding(3)var<storage,read_write>ap:eE;@group(0)@binding(4)var<storage,read_write>bN:array<cJ>;@group(0)@binding(5)var<storage,read_write>W:array<bs>;const o=256u;var<workgroup>be:array<j,o>;var<workgroup>jh:j;var<workgroup>fF:j;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fF=atomicLoad(&ap.at);}let at=workgroupUniformLoad(&fF);if(at&eD)!=e{return;}let aD=d/h(cQ);let bq=d/h(bu);let aH=N.x;var bn=cM;if aH<n.cN{bn=s[n.c6+aH];}var J=0;var S=0;var T=0;var V=0;if bn!=cM&&bn!=ev{let l=hl[aH];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aD));S=m(floor(l.y*bq));T=m(ceil(l.z*aD));V=m(ceil(l.w*bq));}}let fK=j(clamp(J,0,m(n.aM)));let fJ=j(clamp(S,0,m(n.cO)));let fI=j(clamp(T,0,m(n.aM)));let fH=j(clamp(V,0,m(n.cO)));let ca=(fI-fK)*(fH-fJ);var d9=ca;be[k.x]=ca;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){d9+=be[k.x-(f<<i)];}workgroupBarrier();be[k.x]=d9;}if k.x==o-f{let cb=be[o- f];var a5=atomicAdd(&ap.F,cb);if a5+cb>n.iz{a5=e;atomicOr(&ap.at,fz);}bN[aH].W=a5;}storageBarrier();let fG=bN[aH|(o- f)].W;storageBarrier();if aH<n.cN{let hn=select(e,be[k.x- f],k.x>e);let l=vec4(fK,fJ,fI,fH);let X=cJ(l,fG+hn);bN[aH]=X;}let hm=be[o- f];for(var i=k.x;i<hm;i+=o){W[fG+i]=bs(0,e);}}`
