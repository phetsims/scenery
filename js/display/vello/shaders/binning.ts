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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>b7:array<aQ>;@group(0)@binding(2)var<storage>ij:array<da>;@group(0)@binding(3)var<storage>ii:array<af>;@group(0)@binding(4)var<storage,read_write>ih:array<af>;@group(0)@binding(5)var<storage,read_write>aq:ew;@group(0)@binding(6)var<storage,read_write>ig:aX;struct fl{bq:i,ck:i}@group(0)@binding(7)var<storage,read_write>dI:array<fl>;const aD=.00390625;const bs=.00390625;const o=256u;const cl=8u;const gE=4u;var<workgroup>bF:array<array<bY,aN>,cl>;var<workgroup>gy:array<array<i,aN>,gE>;var<workgroup>gx:array<i,aN>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)at:M){for(var j=e;j<cl;j+=f){atomicStore(&bF[j][k.x],e);}workgroupBarrier();let ee=N.x;var J=0;var R=0;var T=0;var V=0;if ee<n.cO{let dK=b7[ee];var gA=vec4(-1e9,-1e9,1e9,1e9);if dK.cn>e{gA=ii[min(dK.cn-f,n.es- f)];}let aU=ij[dK.ac];let io=af(vec4(aU.J,aU.R,aU.T,aU.V));let l=dV(gA,io);ih[ee]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aD));R=m(floor(l.y*bs));T=m(ceil(l.z*aD));V=m(ceil(l.w*bs));}}let cZ=m((n.aM+bi- f)/bi);let gD=m((n.cP+cQ- f)/cQ);J=clamp(J,0,cZ);R=clamp(R,0,gD);T=clamp(T,0,cZ);V=clamp(V,0,gD);if J==T{V=R;}var x=J;var y=R;let ed=k.x/32u;let fk=f<<(k.x&31u);while y<V{atomicOr(&bF[ed][y*cZ+x],fk);x+=1;if x==T{x=J;y+=1;}}workgroupBarrier();var bq=e;for(var j=e;j<gE;j+=f){bq+=countOneBits(atomicLoad(&bF[j*2u][k.x]));let in=bq;bq+=countOneBits(atomicLoad(&bF[j*2u+f][k.x]));let im=bq;let il=in|(im<<16u);gy[j][k.x]=il;}var ck=atomicAdd(&aq.fA,bq);if ck+bq>n.iA{ck=e;atomicOr(&aq.au,ev);}gx[k.x]=ck;dI[N.x].bq=bq;dI[N.x].ck=ck;workgroupBarrier();x=J;y=R;while y<V{let dJ=y*cZ+x;let gC=atomicLoad(&bF[ed][dJ]);if(gC&fk)!=e{var gz=countOneBits(gC&(fk- f));if ed>e{let gB=ed- f;let ik=gy[gB/2u][dJ];gz+=(ik>>(16u*(gB&f)))&65535u;}let a5=n.g4+gx[dJ];ig[a5+gz]=ee;}x+=1;if x==T{x=J;y+=1;}}}`
