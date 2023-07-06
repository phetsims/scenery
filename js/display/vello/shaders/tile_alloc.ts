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
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>u:a0;@group(0)@binding(2)var<storage>hu:array<L>;@group(0)@binding(3)var<storage,read_write>ao:eH;@group(0)@binding(4)var<storage,read_write>bQ:array<cL>;@group(0)@binding(5)var<storage,read_write>Y:array<bw>;const o=256u;var<workgroup>bk:array<j,o>;var<workgroup>jm:j;var<workgroup>fK:j;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){if k.x==e{fK=atomicLoad(&ao.aq);}let aq=workgroupUniformLoad(&fK);if(aq&eG)!=e{return;}let aD=d/h(cT);let bv=d/h(bx);let aH=N.x;var bs=cP;if aH<n.cQ{bs=u[n.c8+aH];}var J=0;var S=0;var T=0;var V=0;if bs!=cP&&bs!=ex{let l=hu[aH];if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aD));S=m(floor(l.y*bv));T=m(ceil(l.z*aD));V=m(ceil(l.w*bv));}}let fP=j(clamp(J,0,m(n.aM)));let fO=j(clamp(S,0,m(n.cR)));let fN=j(clamp(T,0,m(n.aM)));let fM=j(clamp(V,0,m(n.cR)));let cb=(fN-fP)*(fM-fO);var eb=cb;bk[k.x]=cb;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){eb+=bk[k.x-(f<<i)];}workgroupBarrier();bk[k.x]=eb;}if k.x==o-f{let cc=bk[o- f];var a5=atomicAdd(&ao.F,cc);if a5+cc>n.iD{a5=e;atomicOr(&ao.aq,fE);}bQ[aH].Y=a5;}storageBarrier();let fL=bQ[aH|(o- f)].Y;storageBarrier();if aH<n.cQ{let hw=select(e,bk[k.x- f],k.x>e);let l=vec4(fP,fO,fN,fM);let X=cL(l,fL+hw);bQ[aH]=X;}let hv=bk[o- f];for(var i=k.x;i<hv;i+=o){Y[fL+i]=bw(0,e);}}`
