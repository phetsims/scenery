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
@group(0)@binding(0)var<uniform>j:aM;@group(0)@binding(1)var<storage>n:aS;@group(0)@binding(2)var<storage>hh:array<Z>;@group(0)@binding(3)var<storage,read_write>ak:es;@group(0)@binding(4)var<storage,read_write>bM:array<cF>;@group(0)@binding(5)var<storage,read_write>S:array<bn>;const k=256u;var<workgroup>a6:array<d,k>;var<workgroup>jd:d;var<workgroup>fD:d;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)J:I,@builtin(local_invocation_id)f:I){if f.x==0u{fD=atomicLoad(&ak.am);}let am=workgroupUniformLoad(&fD);if(am&er)!=0u{return;}let ay=1./c(cM);let bm=1./c(bq);let aC=J.x;var bj=cI;if aC<j.cJ{bj=n[j.c3+aC];}var F=0;var N=0;var O=0;var R=0;if bj!=cI&&bj!=ej{let h=hh[aC];if h.x<h.z&&h.y<h.w{F=i(floor(h.x*ay));N=i(floor(h.y*bm));O=i(ceil(h.z*ay));R=i(ceil(h.w*bm));}}let fI=d(clamp(F,0,i(j.aH)));let fH=d(clamp(N,0,i(j.cK)));let fG=d(clamp(O,0,i(j.aH)));let fF=d(clamp(R,0,i(j.cK)));let ca=(fG-fI)*(fF-fH);var d2=ca;a6[f.x]=ca;for(var e=0u;e<firstTrailingBit(k);e+=1u){workgroupBarrier();if f.x>=(1u<<e){d2+=a6[f.x-(1u<<e)];}workgroupBarrier();a6[f.x]=d2;}if f.x==k- 1u{let cb=a6[k- 1u];var aZ=atomicAdd(&ak.B,cb);if aZ+cb>j.iv{aZ=0u;atomicOr(&ak.am,fx);}bM[aC].S=aZ;}storageBarrier();let fE=bM[aC|(k- 1u)].S;storageBarrier();if aC<j.cJ{let hj=select(0u,a6[f.x- 1u],f.x>0u);let h=vec4(fI,fH,fG,fF);let T=cF(h,fE+hj);bM[aC]=T;}let hi=a6[k- 1u];for(var e=f.x;e<hi;e+=k){S[fE+e]=bn(0,0u);}}`
