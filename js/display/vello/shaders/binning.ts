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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>b_:array<aQ>;@group(0)@binding(2)var<storage>ik:array<c8>;@group(0)@binding(3)var<storage>ij:array<L>;@group(0)@binding(4)var<storage,read_write>ii:array<L>;@group(0)@binding(5)var<storage,read_write>ao:eD;@group(0)@binding(6)var<storage,read_write>ih:aX;struct fh{bq:j,cd:j}@group(0)@binding(7)var<storage,read_write>dF:array<fh>;const aC=.00390625;const bs=.00390625;const o=256u;const ce=8u;const gE=4u;var<workgroup>bD:array<array<bS,aL>,ce>;var<workgroup>gy:array<array<j,aL>,gE>;var<workgroup>gx:array<j,aL>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){for(var i=e;i<ce;i+=f){atomicStore(&bD[i][k.x],e);}workgroupBarrier();let ek=N.x;var J=0;var S=0;var T=0;var V=0;if ek<n.cN{let dH=b_[ek];var gA=vec4(-1e9,-1e9,1e9,1e9);if dH.cl>e{gA=ij[min(dH.cl-f,n.ez- f)];}let aU=ik[dH.ac];let ip=L(vec4(aU.J,aU.S,aU.T,aU.V));let l=dS(gA,ip);ii[ek]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bs));T=m(ceil(l.z*aC));V=m(ceil(l.w*bs));}}let cY=m((n.aK+bi- f)/bi);let gD=m((n.cO+cP- f)/cP);J=clamp(J,0,cY);S=clamp(S,0,gD);T=clamp(T,0,cY);V=clamp(V,0,gD);if J==T{V=S;}var x=J;var y=S;let ej=k.x/32u;let fg=f<<(k.x&31u);while y<V{atomicOr(&bD[ej][y*cY+x],fg);x+=1;if x==T{x=J;y+=1;}}workgroupBarrier();var bq=e;for(var i=e;i<gE;i+=f){bq+=countOneBits(atomicLoad(&bD[i*2u][k.x]));let io=bq;bq+=countOneBits(atomicLoad(&bD[i*2u+f][k.x]));let in=bq;let im=io|(in<<16u);gy[i][k.x]=im;}var cd=atomicAdd(&ao.fx,bq);if cd+bq>n.iB{cd=e;atomicOr(&ao.aq,eC);}gx[k.x]=cd;dF[N.x].bq=bq;dF[N.x].cd=cd;workgroupBarrier();x=J;y=S;while y<V{let dG=y*cY+x;let gC=atomicLoad(&bD[ej][dG]);if(gC&fg)!=e{var gz=countOneBits(gC&(fg- f));if ej>e{let gB=ej- f;let il=gy[gB/2u][dG];gz+=(il>>(16u*(gB&f)))&65535u;}let a5=n.g6+gx[dG];ih[a5+gz]=ek;}x+=1;if x==T{x=J;y+=1;}}}`
