/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${bbox}
${clip}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>cX:array<eu>;@group(0)@binding(2)var<storage>bR:array<da>;@group(0)@binding(3)var<storage,read_write>aP:array<a3>;@group(0)@binding(4)var<storage,read_write>h8:array<fz>;const o=256u;var<workgroup>aO:array<a3,o>;var<workgroup>bp:array<i,o>;var<workgroup>go:array<i,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)at:M){let cY=cX[N.x].ac;let cj=cY>=0;var am=a3(f-i(cj),i(cj));aO[k.x]=am;for(var j=e;j<firstTrailingBit(o);j+=f){workgroupBarrier();if k.x+(f<<j)<o{let aw=aO[k.x+(f<<j)];am=dU(am,aw);}workgroupBarrier();aO[k.x]=am;}if k.x==e{aP[at.x]=am;}workgroupBarrier();let size=aO[0].b;am=a3();if cj&&am.a==e{let gp=size-am.b-f;bp[gp]=k.x;go[gp]=i(cY);}workgroupBarrier();if k.x<size{let ac=go[k.x];let aU=bR[ac];let dT=bp[k.x]+at.x*o;let l=vec4(h(aU.J),h(aU.R),h(aU.T),h(aU.V));h8[N.x]=fz(dT,l);}}`
