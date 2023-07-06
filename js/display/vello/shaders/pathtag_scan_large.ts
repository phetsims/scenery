/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage>aN:array<ak>;@group(0)@binding(3)var<storage,read_write>b6:array<ak>;const bU=8u;const o=256u;var<workgroup>bT:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let p=N.x;let K=u[n.dQ+p];var dU=et(K);bT[k.x]=dU;for(var i=e;i<bU;i+=f){workgroupBarrier();if k.x>=f<<i{let au=bT[k.x-(f<<i)];dU=bP(au,dU);}workgroupBarrier();bT[k.x]=dU;}workgroupBarrier();var U=aN[ap.x];if k.x>e{U=bP(U,bT[k.x-f]);}b6[p]=U;}`
