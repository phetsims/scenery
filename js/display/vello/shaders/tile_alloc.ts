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
@group(0)@binding(0)var<uniform>n:aT;@group(0)@binding(1)var<storage>u:aZ;@group(0)@binding(2)var<storage>hs:array<L>;@group(0)@binding(3)var<storage,read_write>ao:eG;@group(0)@binding(4)var<storage,read_write>bP:array<cK>;@group(0)@binding(5)var<storage,read_write>X:array<bv>;const o=256u;var<workgroup>bi:array<j,o>;var<workgroup>jk:j;var<workgroup>fI:j;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fI=atomicLoad(&ao.aq);}let aq=workgroupUniformLoad(&fI);if(aq&eF)!=e{return;}let aC=d/h(cS);let bu=d/h(bw);let aH=N.x;var bq=cO;if aH<n.cP{bq=u[n.c7+aH];}var J=0;var S=0;var T=0;var V=0;if bq!=cO&&bq!=ew{let l=hs[aH];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bu));T=m(ceil(l.z*aC));V=m(ceil(l.w*bu));}}let fN=j(clamp(J,0,m(n.aL)));let fM=j(clamp(S,0,m(n.cQ)));let fL=j(clamp(T,0,m(n.aL)));let fK=j(clamp(V,0,m(n.cQ)));let ca=(fL-fN)*(fK-fM);var ea=ca;bi[k.x]=ca;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){ea+=bi[k.x-(f<<i)];}workgroupBarrier();bi[k.x]=ea;}if k.x==o-f{let cb=bi[o- f];var a4=atomicAdd(&ao.F,cb);if a4+cb>n.iC{a4=e;atomicOr(&ao.aq,fC);}bP[aH].X=a4;}storageBarrier();let fJ=bP[aH|(o- f)].X;storageBarrier();if aH<n.cP{let hu=select(e,bi[k.x- f],k.x>e);let l=vec4(fN,fM,fL,fK);let W=cK(l,fJ+hu);bP[aH]=W;}let ht=bi[o- f];for(var i=k.x;i<ht;i+=o){X[fJ+i]=bv(0,e);}}`
