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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>b0:array<aQ>;@group(0)@binding(2)var<storage>in:array<c8>;@group(0)@binding(3)var<storage>im:array<L>;@group(0)@binding(4)var<storage,read_write>il:array<L>;@group(0)@binding(5)var<storage,read_write>ao:eE;@group(0)@binding(6)var<storage,read_write>ik:aX;struct fj{bs:j,cd:j}@group(0)@binding(7)var<storage,read_write>dE:array<fj>;const aC=.00390625;const bt=.00390625;const o=256u;const ce=8u;const gE=4u;var<workgroup>bE:array<array<bT,aL>,ce>;var<workgroup>gy:array<array<j,aL>,gE>;var<workgroup>gx:array<j,aL>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){for(var i=e;i<ce;i+=f){atomicStore(&bE[i][k.x],e);}workgroupBarrier();let ei=N.x;var J=0;var S=0;var T=0;var V=0;if ei<n.cN{let dG=b0[ei];var gA=vec4(-1e9,-1e9,1e9,1e9);if dG.cl>e{gA=im[min(dG.cl-f,n.eA- f)];}let aU=in[dG.ac];let is=L(vec4(aU.J,aU.S,aU.T,aU.V));let l=dR(gA,is);il[ei]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bt));T=m(ceil(l.z*aC));V=m(ceil(l.w*bt));}}let cY=m((n.aK+bj- f)/bj);let gD=m((n.cO+cP- f)/cP);J=clamp(J,0,cY);S=clamp(S,0,gD);T=clamp(T,0,cY);V=clamp(V,0,gD);if J==T{V=S;}var x=J;var y=S;let eh=k.x/32u;let fi=f<<(k.x&31u);while y<V{atomicOr(&bE[eh][y*cY+x],fi);x+=1;if x==T{x=J;y+=1;}}workgroupBarrier();var bs=e;for(var i=e;i<gE;i+=f){bs+=countOneBits(atomicLoad(&bE[i*2u][k.x]));let ir=bs;bs+=countOneBits(atomicLoad(&bE[i*2u+f][k.x]));let iq=bs;let ip=ir|(iq<<16u);gy[i][k.x]=ip;}var cd=atomicAdd(&ao.fx,bs);if cd+bs>n.iD{cd=e;atomicOr(&ao.aq,eD);}gx[k.x]=cd;dE[N.x].bs=bs;dE[N.x].cd=cd;workgroupBarrier();x=J;y=S;while y<V{let dF=y*cY+x;let gC=atomicLoad(&bE[eh][dF]);if(gC&fi)!=e{var gz=countOneBits(gC&(fi- f));if eh>e{let gB=eh- f;let io=gy[gB/2u][dF];gz+=(io>>(16u*(gB&f)))&65535u;}let a7=n.g6+gx[dF];ik[a7+gz]=ei;}x+=1;if x==T{x=J;y+=1;}}}`
