/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>s:aX;@group(0)@binding(2)var<storage>aP:array<ak>;@group(0)@binding(3)var<storage,read_write>b6:array<ak>;const bV=8u;const o=256u;var<workgroup>bU:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)aq:M){let p=N.x;let K=s[n.dR+p];var dV=eu(K);bU[k.x]=dV;for(var i=e;i<bV;i+=f){workgroupBarrier();if k.x>=f<<i{let av=bU[k.x-(f<<i)];dV=bQ(av,dV);}workgroupBarrier();bU[k.x]=dV;}workgroupBarrier();var U=aP[aq.x];if k.x>e{U=bQ(U,bU[k.x-f]);}b6[p]=U;}`
