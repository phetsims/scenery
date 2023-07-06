/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<storage>aN:array<ak>;@group(0)@binding(1)var<storage>hq:array<ak>;@group(0)@binding(2)var<storage,read_write>b5:array<ak>;const bU=8u;const o=256u;var<workgroup>bp:array<ak,o>;var<workgroup>bT:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){var agg=g3();if k.x<ap.x{agg=hq[k.x];}bp[k.x]=agg;for(var i=e;i<bU;i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=bp[k.x+(f<<i)];agg=bO(agg,au);}workgroupBarrier();bp[k.x]=agg;}let p=N.x;agg=aN[p];bT[k.x]=agg;for(var i=e;i<bU;i+=f){workgroupBarrier();if k.x>=f<<i{let au=bT[k.x-(f<<i)];agg=bO(au,agg);}workgroupBarrier();bT[k.x]=agg;}workgroupBarrier();var U=bp[0];if k.x>e{U=bO(U,bT[k.x-f]);}b5[p]=U;}`
