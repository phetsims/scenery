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
@group(0)@binding(0)var<uniform>n:aQ;@group(0)@binding(1)var<storage>s:aW;@group(0)@binding(2)var<storage>hl:array<af>;@group(0)@binding(3)var<storage,read_write>ao:ew;@group(0)@binding(4)var<storage,read_write>bQ:array<cJ>;@group(0)@binding(5)var<storage,read_write>W:array<bs>;const o=256u;var<workgroup>bc:array<i,o>;var<workgroup>jh:i;var<workgroup>fH:i;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fH=atomicLoad(&ao.aq);}let aq=workgroupUniformLoad(&fH);if(aq&ev)!=e{return;}let aC=d/h(cQ);let bq=d/h(bv);let aG=N.x;var bn=cM;if aG<n.cN{bn=s[n.c7+aG];}var J=0;var R=0;var S=0;var V=0;if bn!=cM&&bn!=en{let l=hl[aG];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));R=m(floor(l.y*bq));S=m(ceil(l.z*aC));V=m(ceil(l.w*bq));}}let fM=i(clamp(J,0,m(n.aL)));let fL=i(clamp(R,0,m(n.cO)));let fK=i(clamp(S,0,m(n.aL)));let fJ=i(clamp(V,0,m(n.cO)));let ce=(fK-fM)*(fJ-fL);var d6=ce;bc[k.x]=ce;for(var j=e;j<firstTrailingBit(o);j+=f){workgroupBarrier();if k.x>=(f<<j){d6+=bc[k.x-(f<<j)];}workgroupBarrier();bc[k.x]=d6;}if k.x==o- f{let cf=bc[o- f];var a2=atomicAdd(&ao.F,cf);if a2+cf>n.iz{a2=e;atomicOr(&ao.aq,fB);}bQ[aG].W=a2;}storageBarrier();let fI=bQ[aG|(o- f)].W;storageBarrier();if aG<n.cN{let hn=select(e,bc[k.x- f],k.x>e);let l=vec4(fM,fL,fK,fJ);let X=cJ(l,fI+hn);bQ[aG]=X;}let hm=bc[o- f];for(var j=k.x;j<hm;j+=o){W[fI+j]=bs(0,e);}}`
