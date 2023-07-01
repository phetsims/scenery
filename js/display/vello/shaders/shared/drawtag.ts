/* eslint-disable */

export default `struct _aF{_P:u32,_ce:u32,_by:u32,_N:u32}const _cy=0u;const _fl=0x42u;const _ep=0x10au;const _eo=0x28eu;const _en=0x244u;const _dD=0x2bu;const _em=0x401u;fn _il()->_aF{return _aF();}fn _eq(a:_aF,b:_aF)->_aF{var c:_aF;c._P=a._P+b._P;c._ce=a._ce+b._ce;c._by=a._by+b._by;c._N=a._N+b._N;return c;}fn _gN(_D:u32)->_aF{var c:_aF;c._P=u32(_D!=_cy);c._ce=_D&1u;c._by=(_D>>1u)&0x1fu;c._N=(_D>>6u)&0xfu;return c;}`
