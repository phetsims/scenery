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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>b1:array<aQ>;@group(0)@binding(2)var<storage>ij:array<c8>;@group(0)@binding(3)var<storage>ii:array<L>;@group(0)@binding(4)var<storage,read_write>ih:array<L>;@group(0)@binding(5)var<storage,read_write>ao:eE;@group(0)@binding(6)var<storage,read_write>ig:aX;struct fi{bp:j,ce:j}@group(0)@binding(7)var<storage,read_write>dG:array<fi>;const aC=.00390625;const bq=.00390625;const o=256u;const cf=8u;const gD=4u;var<workgroup>bD:array<array<bS,aL>,cf>;var<workgroup>gx:array<array<j,aL>,gD>;var<workgroup>gw:array<j,aL>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){for(var i=e;i<cf;i+=f){atomicStore(&bD[i][k.x],e);}workgroupBarrier();let el=N.x;var J=0;var S=0;var T=0;var V=0;if el<n.cN{let dI=b1[el];var gz=vec4(-1e9,-1e9,1e9,1e9);if dI.cl>e{gz=ii[min(dI.cl-f,n.eA- f)];}let aU=ij[dI.ac];let io=L(vec4(aU.J,aU.S,aU.T,aU.V));let l=dT(gz,io);ih[el]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bq));T=m(ceil(l.z*aC));V=m(ceil(l.w*bq));}}let cY=m((n.aK+bh- f)/bh);let gC=m((n.cO+cP- f)/cP);J=clamp(J,0,cY);S=clamp(S,0,gC);T=clamp(T,0,cY);V=clamp(V,0,gC);if J==T{V=S;}var x=J;var y=S;let ek=k.x/32u;let fh=f<<(k.x&31u);while y<V{atomicOr(&bD[ek][y*cY+x],fh);x+=1;if x==T{x=J;y+=1;}}workgroupBarrier();var bp=e;for(var i=e;i<gD;i+=f){bp+=countOneBits(atomicLoad(&bD[i*2u][k.x]));let in=bp;bp+=countOneBits(atomicLoad(&bD[i*2u+f][k.x]));let im=bp;let il=in|(im<<16u);gx[i][k.x]=il;}var ce=atomicAdd(&ao.fx,bp);if ce+bp>n.iA{ce=e;atomicOr(&ao.aq,eD);}gw[k.x]=ce;dG[N.x].bp=bp;dG[N.x].ce=ce;workgroupBarrier();x=J;y=S;while y<V{let dH=y*cY+x;let gB=atomicLoad(&bD[ek][dH]);if(gB&fh)!=e{var gy=countOneBits(gB&(fh- f));if ek>e{let gA=ek- f;let ik=gx[gA/2u][dH];gy+=(ik>>(16u*(gA&f)))&65535u;}let a5=n.g4+gw[dH];ig[a5+gy]=el;}x+=1;if x==T{x=J;y+=1;}}}`
