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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>s:aX;@group(0)@binding(2)var<storage>hl:array<af>;@group(0)@binding(3)var<storage,read_write>aq:ew;@group(0)@binding(4)var<storage,read_write>bS:array<cK>;@group(0)@binding(5)var<storage,read_write>W:array<bt>;const o=256u;var<workgroup>be:array<i,o>;var<workgroup>jh:i;var<workgroup>fH:i;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fH=atomicLoad(&aq.au);}let au=workgroupUniformLoad(&fH);if(au&ev)!=e{return;}let aD=d/h(cR);let bs=d/h(bw);let aH=N.x;var bo=cN;if aH<n.cO{bo=s[n.c7+aH];}var J=0;var R=0;var T=0;var V=0;if bo!=cN&&bo!=en{let l=hl[aH];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aD));R=m(floor(l.y*bs));T=m(ceil(l.z*aD));V=m(ceil(l.w*bs));}}let fM=i(clamp(J,0,m(n.aM)));let fL=i(clamp(R,0,m(n.cP)));let fK=i(clamp(T,0,m(n.aM)));let fJ=i(clamp(V,0,m(n.cP)));let cg=(fK-fM)*(fJ-fL);var d6=cg;be[k.x]=cg;for(var j=e;j<firstTrailingBit(o);j+=f){workgroupBarrier();if k.x>=(f<<j){d6+=be[k.x-(f<<j)];}workgroupBarrier();be[k.x]=d6;}if k.x==o-f{let ch=be[o- f];var a5=atomicAdd(&aq.F,ch);if a5+ch>n.iz{a5=e;atomicOr(&aq.au,fB);}bS[aH].W=a5;}storageBarrier();let fI=bS[aH|(o- f)].W;storageBarrier();if aH<n.cO{let hn=select(e,be[k.x- f],k.x>e);let l=vec4(fM,fL,fK,fJ);let X=cK(l,fI+hn);bS[aH]=X;}let hm=be[o- f];for(var j=k.x;j<hm;j+=o){W[fI+j]=bt(0,e);}}`
