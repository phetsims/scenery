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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>b_:array<aQ>;@group(0)@binding(2)var<storage>ij:array<c8>;@group(0)@binding(3)var<storage>ii:array<L>;@group(0)@binding(4)var<storage,read_write>ih:array<L>;@group(0)@binding(5)var<storage,read_write>ao:eD;@group(0)@binding(6)var<storage,read_write>ig:aX;struct fh{bp:j,cc:j}@group(0)@binding(7)var<storage,read_write>dF:array<fh>;const aC=.00390625;const bq=.00390625;const o=256u;const cd=8u;const gC=4u;var<workgroup>bC:array<array<bS,aL>,cd>;var<workgroup>gw:array<array<j,aL>,gC>;var<workgroup>gv:array<j,aL>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){for(var i=e;i<cd;i+=f){atomicStore(&bC[i][k.x],e);}workgroupBarrier();let ek=N.x;var J=0;var S=0;var T=0;var V=0;if ek<n.cN{let dH=b_[ek];var gy=vec4(-1e9,-1e9,1e9,1e9);if dH.cl>e{gy=ii[min(dH.cl-f,n.ez- f)];}let aU=ij[dH.ac];let io=L(vec4(aU.J,aU.S,aU.T,aU.V));let l=dS(gy,io);ih[ek]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bq));T=m(ceil(l.z*aC));V=m(ceil(l.w*bq));}}let cY=m((n.aK+bh- f)/bh);let gB=m((n.cO+cP- f)/cP);J=clamp(J,0,cY);S=clamp(S,0,gB);T=clamp(T,0,cY);V=clamp(V,0,gB);if J==T{V=S;}var x=J;var y=S;let ej=k.x/32u;let fg=f<<(k.x&31u);while y<V{atomicOr(&bC[ej][y*cY+x],fg);x+=1;if x==T{x=J;y+=1;}}workgroupBarrier();var bp=e;for(var i=e;i<gC;i+=f){bp+=countOneBits(atomicLoad(&bC[i*2u][k.x]));let in=bp;bp+=countOneBits(atomicLoad(&bC[i*2u+f][k.x]));let im=bp;let il=in|(im<<16u);gw[i][k.x]=il;}var cc=atomicAdd(&ao.fw,bp);if cc+bp>n.iA{cc=e;atomicOr(&ao.aq,eC);}gv[k.x]=cc;dF[N.x].bp=bp;dF[N.x].cc=cc;workgroupBarrier();x=J;y=S;while y<V{let dG=y*cY+x;let gA=atomicLoad(&bC[ej][dG]);if(gA&fg)!=e{var gx=countOneBits(gA&(fg- f));if ej>e{let gz=ej- f;let ik=gw[gz/2u][dG];gx+=(ik>>(16u*(gz&f)))&65535u;}let a5=n.g4+gv[dG];ig[a5+gx]=ek;}x+=1;if x==T{x=J;y+=1;}}}`
