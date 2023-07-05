/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${bbox}
${clip}
@group(0)@binding(0)var<uniform>n:aQ;@group(0)@binding(1)var<storage>cX:array<eu>;@group(0)@binding(2)var<storage>bP:array<da>;@group(0)@binding(3)var<storage,read_write>aO:array<a0>;@group(0)@binding(4)var<storage,read_write>h8:array<fz>;const o=256u;var<workgroup>aN:array<a0,o>;var<workgroup>bo:array<i,o>;var<workgroup>go:array<i,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let cY=cX[N.x].ac;let ch=cY>=0;var al=a0(f-i(ch),i(ch));aN[k.x]=al;for(var j=e;j<firstTrailingBit(o);j+=f){workgroupBarrier();if k.x+(f<<j)<o{let au=aN[k.x+(f<<j)];al=dU(al,au);}workgroupBarrier();aN[k.x]=al;}if k.x==e{aO[ap.x]=al;}workgroupBarrier();let size=aN[0].b;al=a0();if ch&&al.a==e{let gp=size-al.b- f;bo[gp]=k.x;go[gp]=i(cY);}workgroupBarrier();if k.x<size{let ac=go[k.x];let aT=bP[ac];let dT=bo[k.x]+ap.x*o;let l=vec4(h(aT.J),h(aT.R),h(aT.S),h(aT.V));h8[N.x]=fz(dT,l);}}`
