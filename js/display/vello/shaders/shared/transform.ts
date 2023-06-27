/* eslint-disable */

export default `struct _ay{_z:vec4<f32>,_ba:vec2<f32>}fn _cE(_u:_ay,p:vec2<f32>)->vec2<f32>{return _u._z.xy*p.x+_u._z.zw*p.y+_u._ba;}fn _eY(_u:_ay)->_ay{let _ib=1.0/(_u._z.x*_u._z.w-_u._z.y*_u._z.z);let _eX=_ib*vec4(_u._z.w,-_u._z.y,-_u._z.z,_u._z.x);let _ia=mat2x2(_eX.xy,_eX.zw)*-_u._ba;return _ay(_eX,_ia);}fn _dm(a:_ay,b:_ay)->_ay{return _ay(
a._z.xyxy*b._z.xxzz+a._z.zwzw*b._z.yyww,a._z.xy*b._ba.x+a._z.zw*b._ba.y+a._ba
);}`
