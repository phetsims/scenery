/* eslint-disable */
import bump from './shared/bump.js';
import bbox from './shared/bbox.js';
import drawtag from './shared/drawtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${drawtag}
${bbox}
${bump}
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>b4:array<aU>;@group(0)@binding(2)var<storage>io:array<db>;@group(0)@binding(3)var<storage>in:array<L>;@group(0)@binding(4)var<storage,read_write>im:array<L>;@group(0)@binding(5)var<storage,read_write>ao:eH;@group(0)@binding(6)var<storage,read_write>il:a0;struct fp{bu:j,cf:j}@group(0)@binding(7)var<storage,read_write>dI:array<fp>;const aD=.00390625;const bv=.00390625;const o=256u;const cg=8u;const gL=4u;var<workgroup>bH:array<array<bX,aN>,cg>;var<workgroup>gF:array<array<j,aN>,gL>;var<workgroup>gE:array<j,aN>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){for(var i=e;i<cg;i+=f){atomicStore(&bH[i][k.x],e);}workgroupBarrier();let eo=N.x;var J=0;var S=0;var T=0;var V=0;if eo<n.cQ{let dK=b4[eo];var gH=vec4(-1e9,-1e9,1e9,1e9);if dK.cn>e{gH=in[min(dK.cn-f,n.eD- f)];}let aY=io[dK.ac];let it=L(vec4(aY.J,aY.S,aY.T,aY.V));let l=dV(gH,it);im[eo]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aD));S=m(floor(l.y*bv));T=m(ceil(l.z*aD));V=m(ceil(l.w*bv));}}let c0=m((n.aM+bm- f)/bm);let gK=m((n.cR+cS- f)/cS);J=clamp(J,0,c0);S=clamp(S,0,gK);T=clamp(T,0,c0);V=clamp(V,0,gK);if J==T{V=S;}var x=J;var y=S;let en=k.x/32u;let fo=f<<(k.x&31u);while y<V{atomicOr(&bH[en][y*c0+x],fo);x+=1;if x==T{x=J;y+=1;}}workgroupBarrier();var bu=e;for(var i=e;i<gL;i+=f){bu+=countOneBits(atomicLoad(&bH[i*2u][k.x]));let is=bu;bu+=countOneBits(atomicLoad(&bH[i*2u+f][k.x]));let ir=bu;let iq=is|(ir<<16u);gF[i][k.x]=iq;}var cf=atomicAdd(&ao.fD,bu);if cf+bu>n.iE{cf=e;atomicOr(&ao.aq,eG);}gE[k.x]=cf;dI[N.x].bu=bu;dI[N.x].cf=cf;workgroupBarrier();x=J;y=S;while y<V{let dJ=y*c0+x;let gJ=atomicLoad(&bH[en][dJ]);if(gJ&fo)!=e{var gG=countOneBits(gJ&(fo- f));if en>e{let gI=en- f;let ip=gF[gI/2u][dJ];gG+=(ip>>(16u*(gI&f)))&65535u;}let a5=n.hd+gE[dJ];il[a5+gG]=eo;}x+=1;if x==T{x=J;y+=1;}}}`
