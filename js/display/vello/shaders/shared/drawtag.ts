/* eslint-disable */

export default `struct _aE{_N:u32,
_cq:u32,
_bq:u32,
_K:u32}const _dt=0u;const _fe=0x44u;const _eg=0x114u;const _ef=0x29cu;const _ee=0x248u;const _ds=0x9u;const _ed=0x21u;fn _ic()->_aE{return _aE();}fn _eh(a:_aE,b:_aE)->_aE{var c:_aE;c._N=a._N+b._N;c._cq=a._cq+b._cq;c._bq=a._bq+b._bq;c._K=a._K+b._K;return c;}fn _gE(_D:u32)->_aE{var c:_aE;c._N=u32(_D !=_dt);c._cq=_D&1u;c._bq=(_D>>2u)&0x07u;c._K=(_D>>6u)&0x0fu;return c;}`
