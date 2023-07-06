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
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>u:a0;@group(0)@binding(2)var<storage>ht:array<L>;@group(0)@binding(3)var<storage,read_write>ao:eG;@group(0)@binding(4)var<storage,read_write>bP:array<cK>;@group(0)@binding(5)var<storage,read_write>X:array<bv>;const o=256u;var<workgroup>bj:array<j,o>;var<workgroup>jl:j;var<workgroup>fJ:j;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fJ=atomicLoad(&ao.aq);}let aq=workgroupUniformLoad(&fJ);if(aq&eF)!=e{return;}let aC=d/h(cS);let bu=d/h(bw);let aH=N.x;var bq=cO;if aH<n.cP{bq=u[n.c7+aH];}var J=0;var S=0;var T=0;var V=0;if bq!=cO&&bq!=ew{let l=ht[aH];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bu));T=m(ceil(l.z*aC));V=m(ceil(l.w*bu));}}let fO=j(clamp(J,0,m(n.aM)));let fN=j(clamp(S,0,m(n.cQ)));let fM=j(clamp(T,0,m(n.aM)));let fL=j(clamp(V,0,m(n.cQ)));let ca=(fM-fO)*(fL-fN);var ea=ca;bj[k.x]=ca;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){ea+=bj[k.x-(f<<i)];}workgroupBarrier();bj[k.x]=ea;}if k.x==o-f{let cb=bj[o- f];var a5=atomicAdd(&ao.F,cb);if a5+cb>n.iC{a5=e;atomicOr(&ao.aq,fD);}bP[aH].X=a5;}storageBarrier();let fK=bP[aH|(o- f)].X;storageBarrier();if aH<n.cP{let hv=select(e,bj[k.x- f],k.x>e);let l=vec4(fO,fN,fM,fL);let W=cK(l,fK+hv);bP[aH]=W;}let hu=bj[o- f];for(var i=k.x;i<hu;i+=o){X[fK+i]=bv(0,e);}}`
