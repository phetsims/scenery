/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${bbox}
${clip}
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>cZ:array<eF>;@group(0)@binding(2)var<storage>bP:array<db>;@group(0)@binding(3)var<storage,read_write>aQ:array<a7>;@group(0)@binding(4)var<storage,read_write>ie:array<fC>;const o=256u;var<workgroup>aP:array<a7,o>;var<workgroup>bt:array<j,o>;var<workgroup>gw:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let c_=cZ[N.x].ac;let ce=c_>=0;var am=a7(f-j(ce),j(ce));aP[k.x]=am;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=aP[k.x+(f<<i)];am=dU(am,au);}workgroupBarrier();aP[k.x]=am;}if k.x==e{aQ[ap.x]=am;}workgroupBarrier();let size=aP[0].b;am=a7();if ce&&am.a==e{let gx=size-am.b-f;bt[gx]=k.x;gw[gx]=j(c_);}workgroupBarrier();if k.x<size{let ac=gw[k.x];let aY=bP[ac];let dT=bt[k.x]+ap.x*o;let l=vec4(h(aY.J),h(aY.S),h(aY.T),h(aY.V));ie[N.x]=fC(dT,l);}}`
