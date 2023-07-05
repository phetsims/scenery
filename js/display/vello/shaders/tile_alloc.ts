/* eslint-disable */
import tile from './shared/tile.js';
import drawtag from './shared/drawtag.js';
import bump from './shared/bump.js';
import config from './shared/config.js';

export default `${config}
${bump}
${drawtag}
${tile}
@group(0)@binding(0)var<uniform>f:aF;@group(0)@binding(1)var<storage>k:array<u32>;@group(0)@binding(2)var<storage>g6:array<vec4f>;@group(0)@binding(3)var<storage,read_write>ad:eh;@group(0)@binding(4)var<storage,read_write>bE:array<cw>;@group(0)@binding(5)var<storage,read_write>N:array<be>;const h=256u;var<workgroup>aZ:array<u32,h>;var<workgroup>i2:u32;var<workgroup>fs:u32;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)E:vec3u,@builtin(local_invocation_id)d:vec3u,){if d.x==0u{fs=atomicLoad(&ad.af);}let af=workgroupUniformLoad(&fs);if(af&eg)!=0u{return;}let ap=1./f32(cD);let bd=1./f32(bi);let av=E.x;var a8=cz;if av<f.cA{a8=k[f.cV+av];}var B=0;var I=0;var J=0;var M=0;if a8!=cz&&a8!=d8{let e=g6[av];if e.x<e.z&&e.y<e.w{B=i32(floor(e.x*ap));I=i32(floor(e.y*bd));J=i32(ceil(e.z*ap));M=i32(ceil(e.w*bd));}}let fx=u32(clamp(B,0,i32(f.aA)));let fw=u32(clamp(I,0,i32(f.cB)));let fv=u32(clamp(J,0,i32(f.aA)));let fu=u32(clamp(M,0,i32(f.cB)));let b1=(fv-fx)*(fu-fw);var dS=b1;aZ[d.x]=b1;for(var c=0u;c<firstTrailingBit(h);c+=1u){workgroupBarrier();if d.x>=(1u<<c){dS+=aZ[d.x-(1u<<c)];}workgroupBarrier();aZ[d.x]=dS;}if d.x==h- 1u{let b2=aZ[h- 1u];var aR=atomicAdd(&ad.t,b2);if aR+b2>f.ik{aR=0u;atomicOr(&ad.af,fl);}bE[av].N=aR;}storageBarrier();let ft=bE[av|(h- 1u)].N;storageBarrier();if av<f.cA{let g8=select(0u,aZ[d.x- 1u],d.x>0u);let e=vec4(fx,fw,fv,fu);let O=cw(e,ft+g8);bE[av]=O;}let g7=aZ[h- 1u];for(var c=d.x;c<g7;c+=h){N[ft+c]=be(0,0u);}}`
