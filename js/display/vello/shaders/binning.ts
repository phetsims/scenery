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
@group(0)@binding(0)var<uniform>n:aT;@group(0)@binding(1)var<storage>b3:array<aS>;@group(0)@binding(2)var<storage>in:array<da>;@group(0)@binding(3)var<storage>im:array<L>;@group(0)@binding(4)var<storage,read_write>il:array<L>;@group(0)@binding(5)var<storage,read_write>ao:eG;@group(0)@binding(6)var<storage,read_write>ik:aZ;struct fm{bt:j,ce:j}@group(0)@binding(7)var<storage,read_write>dH:array<fm>;const aC=.00390625;const bu=.00390625;const o=256u;const cf=8u;const gJ=4u;var<workgroup>bG:array<array<bW,aM>,cf>;var<workgroup>gD:array<array<j,aM>,gJ>;var<workgroup>gC:array<j,aM>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){for(var i=e;i<cf;i+=f){atomicStore(&bG[i][k.x],e);}workgroupBarrier();let en=N.x;var J=0;var S=0;var T=0;var V=0;if en<n.cP{let dJ=b3[en];var gF=vec4(-1e9,-1e9,1e9,1e9);if dJ.cm>e{gF=im[min(dJ.cm-f,n.eC- f)];}let aW=in[dJ.ac];let is=L(vec4(aW.J,aW.S,aW.T,aW.V));let l=dU(gF,is);il[en]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bu));T=m(ceil(l.z*aC));V=m(ceil(l.w*bu));}}let c_=m((n.aL+bk- f)/bk);let gI=m((n.cQ+cR- f)/cR);J=clamp(J,0,c_);S=clamp(S,0,gI);T=clamp(T,0,c_);V=clamp(V,0,gI);if J==T{V=S;}var x=J;var y=S;let em=k.x/32u;let fl=f<<(k.x&31u);while y<V{atomicOr(&bG[em][y*c_+x],fl);x+=1;if x==T{x=J;y+=1;}}workgroupBarrier();var bt=e;for(var i=e;i<gJ;i+=f){bt+=countOneBits(atomicLoad(&bG[i*2u][k.x]));let ir=bt;bt+=countOneBits(atomicLoad(&bG[i*2u+f][k.x]));let iq=bt;let ip=ir|(iq<<16u);gD[i][k.x]=ip;}var ce=atomicAdd(&ao.fB,bt);if ce+bt>n.iD{ce=e;atomicOr(&ao.aq,eF);}gC[k.x]=ce;dH[N.x].bt=bt;dH[N.x].ce=ce;workgroupBarrier();x=J;y=S;while y<V{let dI=y*c_+x;let gH=atomicLoad(&bG[em][dI]);if(gH&fl)!=e{var gE=countOneBits(gH&(fl- f));if em>e{let gG=em- f;let io=gD[gG/2u][dI];gE+=(io>>(16u*(gG&f)))&65535u;}let a4=n.hb+gC[dI];ik[a4+gE]=en;}x+=1;if x==T{x=J;y+=1;}}}`
