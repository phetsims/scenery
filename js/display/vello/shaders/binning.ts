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
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>b3:array<aU>;@group(0)@binding(2)var<storage>in:array<da>;@group(0)@binding(3)var<storage>im:array<L>;@group(0)@binding(4)var<storage,read_write>il:array<L>;@group(0)@binding(5)var<storage,read_write>ao:eG;@group(0)@binding(6)var<storage,read_write>ik:a0;struct fo{bt:j,ce:j}@group(0)@binding(7)var<storage,read_write>dH:array<fo>;const aC=.00390625;const bu=.00390625;const o=256u;const cf=8u;const gK=4u;var<workgroup>bG:array<array<bW,aN>,cf>;var<workgroup>gE:array<array<j,aN>,gK>;var<workgroup>gD:array<j,aN>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){for(var i=e;i<cf;i+=f){atomicStore(&bG[i][k.x],e);}workgroupBarrier();let en=N.x;var J=0;var S=0;var T=0;var V=0;if en<n.cP{let dJ=b3[en];var gG=vec4(-1e9,-1e9,1e9,1e9);if dJ.cm>e{gG=im[min(dJ.cm-f,n.eC- f)];}let aY=in[dJ.ac];let is=L(vec4(aY.J,aY.S,aY.T,aY.V));let l=dU(gG,is);il[en]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bu));T=m(ceil(l.z*aC));V=m(ceil(l.w*bu));}}let c_=m((n.aM+bl- f)/bl);let gJ=m((n.cQ+cR- f)/cR);J=clamp(J,0,c_);S=clamp(S,0,gJ);T=clamp(T,0,c_);V=clamp(V,0,gJ);if J==T{V=S;}var x=J;var y=S;let em=k.x/32u;let fm=f<<(k.x&31u);while y<V{atomicOr(&bG[em][y*c_+x],fm);x+=1;if x==T{x=J;y+=1;}}workgroupBarrier();var bt=e;for(var i=e;i<gK;i+=f){bt+=countOneBits(atomicLoad(&bG[i*2u][k.x]));let ir=bt;bt+=countOneBits(atomicLoad(&bG[i*2u+f][k.x]));let iq=bt;let ip=ir|(iq<<16u);gE[i][k.x]=ip;}var ce=atomicAdd(&ao.fC,bt);if ce+bt>n.iD{ce=e;atomicOr(&ao.aq,eF);}gD[k.x]=ce;dH[N.x].bt=bt;dH[N.x].ce=ce;workgroupBarrier();x=J;y=S;while y<V{let dI=y*c_+x;let gI=atomicLoad(&bG[em][dI]);if(gI&fm)!=e{var gF=countOneBits(gI&(fm- f));if em>e{let gH=em- f;let io=gE[gH/2u][dI];gF+=(io>>(16u*(gH&f)))&65535u;}let a5=n.hc+gD[dI];ik[a5+gF]=en;}x+=1;if x==T{x=J;y+=1;}}}`
