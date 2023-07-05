/* eslint-disable */

export default `struct aE{R:u32,b8:u32,by:u32,H:u32}const cz=0u;const fg=66u;const eb=0x10au;const ea=0x28eu;const d9=580u;const dC=0x2bu;const d8=1025u;fn ig()->aE{return aE();}fn ec(a:aE,b:aE)->aE{var m:aE;m.R=a.R+b.R;m.b8=a.b8+b.b8;m.by=a.by+b.by;m.H=a.H+b.H;return m;}fn gP(D:u32)->aE{var m:aE;m.R=u32(D!=cz);m.b8=D&1u;m.by=(D>>1u)&0x1fu;m.H=(D>>6u)&0xfu;return m;}`
