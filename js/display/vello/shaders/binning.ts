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
@group(0)@binding(0)var<uniform>n:aQ;@group(0)@binding(1)var<storage>b4:array<aP>;@group(0)@binding(2)var<storage>ij:array<da>;@group(0)@binding(3)var<storage>ii:array<af>;@group(0)@binding(4)var<storage,read_write>ih:array<af>;@group(0)@binding(5)var<storage,read_write>ao:ew;@group(0)@binding(6)var<storage,read_write>ig:aW;struct fl{bp:i,ci:i}@group(0)@binding(7)var<storage,read_write>dI:array<fl>;const aC=.00390625;const bq=.00390625;const o=256u;const cj=8u;const gE=4u;var<workgroup>bE:array<array<bW,aM>,cj>;var<workgroup>gy:array<array<i,aM>,gE>;var<workgroup>gx:array<i,aM>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){for(var j=e;j<cj;j+=f){atomicStore(&bE[j][k.x],e);}workgroupBarrier();let ee=N.x;var J=0;var R=0;var S=0;var V=0;if ee<n.cN{let dK=b4[ee];var gA=vec4(-1e9,-1e9,1e9,1e9);if dK.cl>e{gA=ii[min(dK.cl-f,n.es- f)];}let aT=ij[dK.ac];let io=af(vec4(aT.J,aT.R,aT.S,aT.V));let l=dV(gA,io);ih[ee]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));R=m(floor(l.y*bq));S=m(ceil(l.z*aC));V=m(ceil(l.w*bq));}}let cZ=m((n.aL+bf- f)/bf);let gD=m((n.cO+cP- f)/cP);J=clamp(J,0,cZ);R=clamp(R,0,gD);S=clamp(S,0,cZ);V=clamp(V,0,gD);if J==S{V=R;}var x=J;var y=R;let ed=k.x/32u;let fk=f<<(k.x&31u);while y<V{atomicOr(&bE[ed][y*cZ+x],fk);x+=1;if x==S{x=J;y+=1;}}workgroupBarrier();var bp=e;for(var j=e;j<gE;j+=f){bp+=countOneBits(atomicLoad(&bE[j*2u][k.x]));let in=bp;bp+=countOneBits(atomicLoad(&bE[j*2u+f][k.x]));let im=bp;let il=in|(im<<16u);gy[j][k.x]=il;}var ci=atomicAdd(&ao.fA,bp);if ci+bp>n.iA{ci=e;atomicOr(&ao.aq,ev);}gx[k.x]=ci;dI[N.x].bp=bp;dI[N.x].ci=ci;workgroupBarrier();x=J;y=R;while y<V{let dJ=y*cZ+x;let gC=atomicLoad(&bE[ed][dJ]);if(gC&fk)!=e{var gz=countOneBits(gC&(fk- f));if ed>e{let gB=ed- f;let ik=gy[gB/2u][dJ];gz+=(ik>>(16u*(gB&f)))&65535u;}let a2=n.g4+gx[dJ];ig[a2+gz]=ee;}x+=1;if x==S{x=J;y+=1;}}}`
