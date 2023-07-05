/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>bS:array<cK>;@group(0)@binding(2)var<storage,read_write>W:array<bt>;const o=256u;var<workgroup>gG:array<i,o>;var<workgroup>c_:array<i,o>;var<workgroup>gF:array<i,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let aH=N.x;var eg=e;if aH<n.cO{let X=bS[aH];gG[k.x]=X.l.z-X.l.x;eg=X.l.w-X.l.y;gF[k.x]=X.W;}c_[k.x]=eg;for(var j=e;j<firstTrailingBit(o);j+=f){workgroupBarrier();if k.x>=(f<<j){eg+=c_[k.x-(f<<j)];}workgroupBarrier();c_[k.x]=eg;}workgroupBarrier();let ip=c_[o-f];for(var c0=k.x;c0<ip;c0+=o){var ax=e;for(var j=e;j<firstTrailingBit(o);j+=f){let bG=ax+((o/2u)>>j);if c0>=c_[bG- f]{ax=bG;}}let cm=gG[ax];if cm>e{var ef=c0-select(e,c_[ax- f],ax>e);var aK=gF[ax]+ef*cm;var a6=W[aK].O;for(var x=f;x<cm;x+=f){aK+=f;a6+=W[aK].O;W[aK].O=a6;}}}}`
