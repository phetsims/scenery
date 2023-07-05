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
@group(0)@binding(0)var<uniform>j:aM;@group(0)@binding(1)var<storage>b0:array<aL>;@group(0)@binding(2)var<storage>ie:array<c6>;@group(0)@binding(3)var<storage>ic:array<Z>;@group(0)@binding(4)var<storage,read_write>ib:array<Z>;@group(0)@binding(5)var<storage,read_write>ak:es;@group(0)@binding(6)var<storage,read_write>ia:aS;struct fh{bl:d,ce:d}@group(0)@binding(7)var<storage,read_write>dE:array<fh>;const ay=.00390625;const bm=.00390625;const k=256u;const cf=8u;const gA=4u;var<workgroup>bA:array<array<bS,aI>,cf>;var<workgroup>gu:array<array<d,aI>,gA>;var<workgroup>gt:array<d,aI>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)J:I,@builtin(local_invocation_id)f:I,@builtin(workgroup_id)al:I){for(var e=0u;e<cf;e+=1u){atomicStore(&bA[e][f.x],0u);}workgroupBarrier();let ea=J.x;var F=0;var N=0;var O=0;var R=0;if ea<j.cJ{let dG=b0[ea];var gw=vec4(-1e9,-1e9,1e9,1e9);if dG.ch>0u{gw=ic[min(dG.ch- 1u,j.eo- 1u)];}let aP=ie[dG.W];let ik=Z(vec4(aP.F,aP.N,aP.O,aP.R));let h=dR(gw,ik);ib[ea]=h;if h.x<h.z&&h.y<h.w{F=i(floor(h.x*ay));N=i(floor(h.y*bm));O=i(ceil(h.z*ay));R=i(ceil(h.w*bm));}}let cV=i((j.aH+a9- 1u)/a9);let gz=i((j.cK+cL- 1u)/cL);F=clamp(F,0,cV);N=clamp(N,0,gz);O=clamp(O,0,cV);R=clamp(R,0,gz);if F==O{R=N;}var x=F;var y=N;let d9=f.x/32u;let fg=1u<<(f.x&31u);while y<R{atomicOr(&bA[d9][y*cV+x],fg);x+=1;if x==O{x=F;y+=1;}}workgroupBarrier();var bl=0u;for(var e=0u;e<gA;e+=1u){bl+=countOneBits(atomicLoad(&bA[e*2u][f.x]));let ij=bl;bl+=countOneBits(atomicLoad(&bA[e*2u+1u][f.x]));let ii=bl;let ih=ij|(ii<<16u);gu[e][f.x]=ih;}var ce=atomicAdd(&ak.fw,bl);if ce+bl>j.iw{ce=0u;atomicOr(&ak.am,er);}gt[f.x]=ce;dE[J.x].bl=bl;dE[J.x].ce=ce;workgroupBarrier();x=F;y=N;while y<R{let dF=y*cV+x;let gy=atomicLoad(&bA[d9][dF]);if(gy&fg)!=0u{var gv=countOneBits(gy&(fg- 1u));if d9>0u{let gx=d9- 1u;let ig=gu[gx/2u][dF];gv+=(ig>>(16u*(gx&1u)))&65535u;}let aZ=j.g0+gt[dF];ia[aZ+gv]=ea;}x+=1;if x==O{x=F;y+=1;}}}`
