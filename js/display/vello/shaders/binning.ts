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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>b_:array<aQ>;@group(0)@binding(2)var<storage>ij:array<c8>;@group(0)@binding(3)var<storage>ii:array<L>;@group(0)@binding(4)var<storage,read_write>ih:array<L>;@group(0)@binding(5)var<storage,read_write>ao:eC;@group(0)@binding(6)var<storage,read_write>ig:aX;struct fg{bq:j,cc:j}@group(0)@binding(7)var<storage,read_write>dE:array<fg>;const aC=.00390625;const bs=.00390625;const o=256u;const cd=8u;const gC=4u;var<workgroup>bD:array<array<bS,aL>,cd>;var<workgroup>gw:array<array<j,aL>,gC>;var<workgroup>gv:array<j,aL>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){for(var i=e;i<cd;i+=f){atomicStore(&bD[i][k.x],e);}workgroupBarrier();let ej=N.x;var J=0;var S=0;var T=0;var V=0;if ej<n.cN{let dG=b_[ej];var gy=vec4(-1e9,-1e9,1e9,1e9);if dG.ck>e{gy=ii[min(dG.ck-f,n.ey- f)];}let aU=ij[dG.ac];let io=L(vec4(aU.J,aU.S,aU.T,aU.V));let l=dR(gy,io);ih[ej]=l;if l.x<l.z&&l.y<l.w{J=m(floor(l.x*aC));S=m(floor(l.y*bs));T=m(ceil(l.z*aC));V=m(ceil(l.w*bs));}}let cY=m((n.aK+bi- f)/bi);let gB=m((n.cO+cP- f)/cP);J=clamp(J,0,cY);S=clamp(S,0,gB);T=clamp(T,0,cY);V=clamp(V,0,gB);if J==T{V=S;}var x=J;var y=S;let ei=k.x/32u;let ff=f<<(k.x&31u);while y<V{atomicOr(&bD[ei][y*cY+x],ff);x+=1;if x==T{x=J;y+=1;}}workgroupBarrier();var bq=e;for(var i=e;i<gC;i+=f){bq+=countOneBits(atomicLoad(&bD[i*2u][k.x]));let in=bq;bq+=countOneBits(atomicLoad(&bD[i*2u+f][k.x]));let im=bq;let il=in|(im<<16u);gw[i][k.x]=il;}var cc=atomicAdd(&ao.fv,bq);if cc+bq>n.iA{cc=e;atomicOr(&ao.aq,eB);}gv[k.x]=cc;dE[N.x].bq=bq;dE[N.x].cc=cc;workgroupBarrier();x=J;y=S;while y<V{let dF=y*cY+x;let gA=atomicLoad(&bD[ei][dF]);if(gA&ff)!=e{var gx=countOneBits(gA&(ff- f));if ei>e{let gz=ei- f;let ik=gw[gz/2u][dF];gx+=(ik>>(16u*(gz&f)))&65535u;}let a5=n.g5+gv[dF];ig[a5+gx]=ej;}x+=1;if x==T{x=J;y+=1;}}}`
