/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';

export default `${config}
${tile}
@group(0)@binding(0)var<uniform>f:aF;@group(0)@binding(1)var<storage>bE:array<cw>;@group(0)@binding(2)var<storage,read_write>N:array<be>;const h=256u;var<workgroup>gq:array<u32,h>;var<workgroup>cN:array<u32,h>;var<workgroup>gp:array<u32,h>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)E:vec3u,@builtin(local_invocation_id)d:vec3u,){let av=E.x;var d1=0u;if av<f.cA{let O=bE[av];gq[d.x]=O.e.z-O.e.x;d1=O.e.w-O.e.y;gp[d.x]=O.N;}cN[d.x]=d1;for(var c=0u;c<firstTrailingBit(h);c+=1u){workgroupBarrier();if d.x>=(1u<<c){d1+=cN[d.x-(1u<<c)];}workgroupBarrier();cN[d.x]=d1;}workgroupBarrier();let h8=cN[h- 1u];for(var cO=d.x;cO<h8;cO+=h){var aj=0u;for(var c=0u;c<firstTrailingBit(h);c+=1u){let bt=aj+((h/2u)>>c);if cO>=cN[bt- 1u]{aj=bt;}}let b7=gq[aj];if b7>0u{var d0=cO-select(0u,cN[aj- 1u],aj>0u);var ay=gp[aj]+d0*b7;var aS=N[ay].F;for(var x=1u;x<b7;x+=1u){ay+=1u;aS+=N[ay].F;N[ay].F=aS;}}}}`
