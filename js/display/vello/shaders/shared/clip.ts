/* eslint-disable */

export default `struct Bic{a:u32,b:u32,}fn bic_combine(x:Bic,y:Bic)->Bic{let m=min(x.b,y.a);return Bic(x.a+y.a-m,x.b+y.b-m);}struct ClipInp{ix:u32,


path_ix:i32,}struct ClipEl{parent_ix:u32,bbox:vec4<f32>,}`
