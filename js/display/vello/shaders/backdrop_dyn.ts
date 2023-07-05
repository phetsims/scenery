/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>n:aQ;@group(0)@binding(1)var<storage>bQ:array<cJ>;@group(0)@binding(2)var<storage,read_write>W:array<bs>;const o=256u;var<workgroup>gG:array<i,o>;var<workgroup>c_:array<i,o>;var<workgroup>gF:array<i,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let aG=N.x;var eg=e;if aG<n.cN{let X=bQ[aG];gG[k.x]=X.l.z-X.l.x;eg=X.l.w-X.l.y;gF[k.x]=X.W;}c_[k.x]=eg;for(var j=e;j<firstTrailingBit(o);j+=f){workgroupBarrier();if k.x>=(f<<j){eg+=c_[k.x-(f<<j)];}workgroupBarrier();c_[k.x]=eg;}workgroupBarrier();let ip=c_[o- f];for(var c0=k.x;c0<ip;c0+=o){var av=e;for(var j=e;j<firstTrailingBit(o);j+=f){let bF=av+((o/2u)>>j);if c0>=c_[bF- f]{av=bF;}}let ck=gG[av];if ck>e{var ef=c0-select(e,c_[av- f],av>e);var aJ=gF[av]+ef*ck;var a3=W[aJ].O;for(var x=f;x<ck;x+=f){aJ+=f;a3+=W[aJ].O;W[aJ].O=a3;}}}}`
