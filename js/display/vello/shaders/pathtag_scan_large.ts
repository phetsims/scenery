/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>aN:array<ak>;@group(0)@binding(3)var<storage,read_write>b6:array<ak>;const bV=8u;const o=256u;var<workgroup>bU:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let p=N.x;let K=u[n.dO+p];var dS=et(K);bU[k.x]=dS;for(var i=e;i<bV;i+=f){workgroupBarrier();if k.x>=f<<i{let au=bU[k.x-(f<<i)];dS=bP(au,dS);}workgroupBarrier();bU[k.x]=dS;}workgroupBarrier();var U=aN[ap.x];if k.x>e{U=bP(U,bU[k.x-f]);}b6[p]=U;}`
