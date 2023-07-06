/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${bbox}
${clip}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>cW:array<eC>;@group(0)@binding(2)var<storage>bM:array<c9>;@group(0)@binding(3)var<storage,read_write>aP:array<a3>;@group(0)@binding(4)var<storage,read_write>h8:array<fx>;const o=256u;var<workgroup>aO:array<a3,o>;var<workgroup>bo:array<j,o>;var<workgroup>go:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)aq:M){let cX=cW[N.x].ac;let cd=cX>=0;var al=a3(f-j(cd),j(cd));aO[k.x]=al;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let av=aO[k.x+(f<<i)];al=dT(al,av);}workgroupBarrier();aO[k.x]=al;}if k.x==e{aP[aq.x]=al;}workgroupBarrier();let size=aO[0].b;al=a3();if cd&&al.a==e{let gp=size-al.b-f;bo[gp]=k.x;go[gp]=j(cX);}workgroupBarrier();if k.x<size{let ac=go[k.x];let aU=bM[ac];let dS=bo[k.x]+aq.x*o;let l=vec4(h(aU.J),h(aU.S),h(aU.T),h(aU.V));h8[N.x]=fx(dS,l);}}`
