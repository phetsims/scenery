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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>b2:array<aQ>;@group(0)@binding(2)var<storage>ij:array<c9>;@group(0)@binding(3)var<storage>ii:array<L>;@group(0)@binding(4)var<storage,read_write>ih:array<L>;@group(0)@binding(5)var<storage,read_write>ap:eE;@group(0)@binding(6)var<storage,read_write>ig:aX;struct fj{bp:j,ce:j}@group(0)@binding(7)var<storage,read_write>dH:array<fj>;const aD=.00390625;const bq=.00390625;const o=256u;const cf=8u;const gE=4u;var<workgroup>bD:array<array<bT,aN>,cf>;var<workgroup>gy:array<array<j,aN>,gE>;var<workgroup>gx:array<j,aN>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)aq:M){for(var i=e;i<cf;i+=f){atomicStore(&bD[i][k.x],e);}workgroupBarrier();let em=N.x;var J=0;var S=0;var T=0;var V=0;if em<n.cN{let dJ=b2[em];var gA=vec4(-1e9,-1e9,1e9,1e9);if dJ.cl>e{gA=ii[min(dJ.cl-f,n.eA- f)];}let aU=ij[dJ.ac];let io=L(vec4(aU.J,aU.S,aU.T,aU.V));let l=dU(gA,io);ih[em]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aD));S=m(floor(l.y*bq));T=m(ceil(l.z*aD));V=m(ceil(l.w*bq));}}let cY=m((n.aM+bh- f)/bh);let gD=m((n.cO+cP- f)/cP);J=clamp(J,0,cY);S=clamp(S,0,gD);T=clamp(T,0,cY);V=clamp(V,0,gD);if J==T{V=S;}var x=J;var y=S;let el=k.x/32u;let fi=f<<(k.x&31u);while y<V{atomicOr(&bD[el][y*cY+x],fi);x+=1;if x==T{x=J;y+=1;}}workgroupBarrier();var bp=e;for(var i=e;i<gE;i+=f){bp+=countOneBits(atomicLoad(&bD[i*2u][k.x]));let in=bp;bp+=countOneBits(atomicLoad(&bD[i*2u+f][k.x]));let im=bp;let il=in|(im<<16u);gy[i][k.x]=il;}var ce=atomicAdd(&ap.fy,bp);if ce+bp>n.iA{ce=e;atomicOr(&ap.at,eD);}gx[k.x]=ce;dH[N.x].bp=bp;dH[N.x].ce=ce;workgroupBarrier();x=J;y=S;while y<V{let dI=y*cY+x;let gC=atomicLoad(&bD[el][dI]);if(gC&fi)!=e{var gz=countOneBits(gC&(fi- f));if el>e{let gB=el- f;let ik=gy[gB/2u][dI];gz+=(ik>>(16u*(gB&f)))&65535u;}let a5=n.g4+gx[dI];ig[a5+gz]=em;}x+=1;if x==T{x=J;y+=1;}}}`
