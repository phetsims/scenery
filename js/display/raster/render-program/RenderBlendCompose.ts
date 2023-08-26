// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for binary color-blending and Porter-Duff composition.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderBlendType, RenderColor, RenderComposeType, RenderPath, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import Vector3 from '../../../../../dot/js/Vector3.js';

export default class RenderBlendCompose extends RenderProgram {
  public constructor(
    public readonly composeType: RenderComposeType,
    public readonly blendType: RenderBlendType,
    public readonly a: RenderProgram,
    public readonly b: RenderProgram
  ) {
    super();
  }

  public override getName(): string {
    return 'RenderBlendCompose';
  }

  public override getChildren(): RenderProgram[] {
    return [ this.a, this.b ];
  }

  public override withChildren( children: RenderProgram[] ): RenderBlendCompose {
    assert && assert( children.length === 2 );
    return new RenderBlendCompose( this.composeType, this.blendType, children[ 0 ], children[ 1 ] );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderBlendCompose &&
           this.composeType === other.composeType &&
           this.blendType === other.blendType &&
           this.a.equals( other.a ) &&
           this.b.equals( other.b );
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    // a OP b
    const a = this.a.simplify( pathTest );
    const b = this.b.simplify( pathTest );

    const aTransparent = a.isFullyTransparent();
    const aOpaque = a.isFullyOpaque();
    const bTransparent = b.isFullyTransparent();
    const bOpaque = b.isFullyOpaque();

    if ( aTransparent && bTransparent ) {
      return RenderColor.TRANSPARENT;
    }

    // e.g. only A's contribution
    const mixNormal = this.blendType === RenderBlendType.Normal;

    // TODO: review these, there was a bug in atop!
    // over: fa: 1,   fb: 1-a   fa*a: a      fb*b: b(1-a) sum: a + b(1-a)
    // in:   fa: b,   fb: 0     fa*a: ab     fb*b: 0      sum: ab
    // out:  fa: 1-b, fb: 0     fa*a: a(1-b) fb*b: 0      sum: a(1-b)
    // atop: fa: b,   fb: 1-a   fa*a: ab     fb*b: b(1-a) sum: b
    // xor:  fa: 1-b, fb: 1-a   fa*a: a(1-b) fb*b: b(1-a) sum: a(1-b) + b(1-a)
    // plus: fa: 1,   fb: 1     fa*a: a      fb*b: b      sum: a + b
    // ... plusLighter: NOT THIS
    switch( this.composeType ) {
      case RenderComposeType.Over:
        if ( mixNormal && aOpaque ) {
          return a;
        }
        if ( aTransparent ) {
          return b;
        }
        break;
      case RenderComposeType.In:
        if ( aTransparent || bTransparent ) {
          return RenderColor.TRANSPARENT;
        }
        if ( mixNormal && bOpaque ) {
          return a;
        }
        break;
      case RenderComposeType.Out:
        if ( aTransparent || bOpaque ) {
          return RenderColor.TRANSPARENT;
        }
        if ( mixNormal && bTransparent ) {
          return a;
        }
        break;
      case RenderComposeType.Atop:
        if ( bTransparent ) {
          return RenderColor.TRANSPARENT;
        }
        if ( mixNormal && aOpaque && bOpaque ) {
          return a;
        }
        if ( aTransparent ) {
          return b;
        }
        break;
      case RenderComposeType.Xor:
        if ( aOpaque && bOpaque ) {
          return RenderColor.TRANSPARENT;
        }
        if ( aTransparent ) {
          return b;
        }
        if ( bTransparent ) {
          return a;
        }
        break;
      case RenderComposeType.Plus:
      case RenderComposeType.PlusLighter:
        if ( aTransparent ) {
          return b;
        }
        if ( bTransparent ) {
          return a;
        }
        break;
      default:
        break;
    }

    if ( a instanceof RenderColor && b instanceof RenderColor ) {
      return new RenderColor( RenderBlendCompose.blendCompose( a.color, b.color, this.composeType, this.blendType ) );
    }

    return new RenderBlendCompose( this.composeType, this.blendType, a, b );
  }

  public override isFullyTransparent(): boolean {
    const aTransparent = this.a.isFullyTransparent();
    const aOpaque = this.a.isFullyOpaque();
    const bTransparent = this.b.isFullyTransparent();
    const bOpaque = this.b.isFullyOpaque();

    if ( aTransparent && bTransparent ) {
      return true;
    }

    // over: fa: 1,   fb: 1-a   fa*a: a      fb*b: b(1-a) sum: a + b(1-a)
    // in:   fa: b,   fb: 0     fa*a: ab     fb*b: 0      sum: ab
    // out:  fa: 1-b, fb: 0     fa*a: a(1-b) fb*b: 0      sum: a(1-b)
    // atop: fa: b,   fb: 1-a   fa*a: ab     fb*b: b(1-a) sum: b
    // xor:  fa: 1-b, fb: 1-a   fa*a: a(1-b) fb*b: b(1-a) sum: a(1-b) + b(1-a)
    // plus: fa: 1,   fb: 1     fa*a: a      fb*b: b      sum: a + b
    switch( this.composeType ) {
      case RenderComposeType.In:
        return aTransparent || bTransparent;
      case RenderComposeType.Out:
        return aTransparent || bOpaque;
      case RenderComposeType.Atop:
        return bTransparent;
      case RenderComposeType.Xor:
        return aOpaque && bOpaque;
      default:
        return false;
    }
  }

  public override isFullyOpaque(): boolean {
    const aTransparent = this.a.isFullyTransparent();
    const aOpaque = this.a.isFullyOpaque();
    const bTransparent = this.b.isFullyTransparent();
    const bOpaque = this.b.isFullyOpaque();

    if ( aTransparent && bTransparent ) {
      return false;
    }

    // over: fa: 1,   fb: 1-a   fa*a: a      fb*b: b(1-a) sum: a + b(1-a)
    // in:   fa: b,   fb: 0     fa*a: ab     fb*b: 0      sum: ab
    // out:  fa: 1-b, fb: 0     fa*a: a(1-b) fb*b: 0      sum: a(1-b)
    // atop: fa: b,   fb: 1-a   fa*a: ab     fb*b: b(1-a) sum: b
    // xor:  fa: 1-b, fb: 1-a   fa*a: a(1-b) fb*b: b(1-a) sum: a(1-b) + b(1-a)
    // plus: fa: 1,   fb: 1     fa*a: a      fb*b: b      sum: a + b
    switch( this.composeType ) {
      case RenderComposeType.Over:
        return aOpaque || bOpaque;
      case RenderComposeType.In:
        return aOpaque && bOpaque;
      case RenderComposeType.Out:
        return aOpaque && bTransparent;
      case RenderComposeType.Atop:
        return bOpaque;
      case RenderComposeType.Xor:
        return ( aOpaque && bTransparent ) || ( aTransparent && bOpaque );
      case RenderComposeType.Plus:
        return aOpaque && bOpaque;
      case RenderComposeType.PlusLighter:
        return aOpaque && bOpaque;
      default:
        return false;
    }
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    pathTest: ( renderPath: RenderPath ) => boolean = constantTrue
  ): Vector4 {
    const a = this.a.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );
    const b = this.b.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );

    return RenderBlendCompose.blendCompose( a, b, this.composeType, this.blendType );
  }

  protected override getExtraDebugString(): string {
    return `${RenderComposeType[ this.composeType ]}, ${RenderBlendType[ this.blendType ]}`;
  }

  public static blendCompose( a: Vector4, b: Vector4, composeType: RenderComposeType, blendType: RenderBlendType ): Vector4 {
    let blended;
    if ( blendType === RenderBlendType.Normal ) {
      blended = a;
    }
    else {
      // Need to apply blending when not premultiplied
      const a3 = RenderColor.unpremultiply( a ).toVector3();
      const b3 = RenderColor.unpremultiply( b ).toVector3();
      let c3: Vector3;

      switch( blendType ) {
        case RenderBlendType.Multiply: {
          c3 = b3.componentTimes( a3 );
          break;
        }
        case RenderBlendType.Screen: {
          c3 = RenderBlendCompose.screen( b3, a3 );
          break;
        }
        case RenderBlendType.Overlay: {
          c3 = RenderBlendCompose.hardLight( a3, b3 );
          break;
        }
        case RenderBlendType.Darken: {
          c3 = new Vector3(
            Math.min( b3.x, a3.x ),
            Math.min( b3.y, a3.y ),
            Math.min( b3.z, a3.z )
          );
          break;
        }
        case RenderBlendType.Lighten: {
          c3 = new Vector3(
            Math.max( b3.x, a3.x ),
            Math.max( b3.y, a3.y ),
            Math.max( b3.z, a3.z )
          );
          break;
        }
        case RenderBlendType.ColorDodge: {
          c3 = new Vector3(
            RenderBlendCompose.colorDodge( b3.x, a3.x ),
            RenderBlendCompose.colorDodge( b3.y, a3.y ),
            RenderBlendCompose.colorDodge( b3.z, a3.z )
          );
          break;
        }
        case RenderBlendType.ColorBurn: {
          c3 = new Vector3(
            RenderBlendCompose.colorBurn( b3.x, a3.x ),
            RenderBlendCompose.colorBurn( b3.y, a3.y ),
            RenderBlendCompose.colorBurn( b3.z, a3.z )
          );
          break;
        }
        case RenderBlendType.HardLight: {
          c3 = RenderBlendCompose.hardLight( b3, a3 );
          break;
        }
        case RenderBlendType.SoftLight: {
          c3 = RenderBlendCompose.softLight( b3, a3 );
          break;
        }
        case RenderBlendType.Difference: {
          c3 = new Vector3(
            Math.abs( b3.x - a3.x ),
            Math.abs( b3.y - a3.y ),
            Math.abs( b3.z - a3.z )
          );
          break;
        }
        case RenderBlendType.Exclusion: {
          c3 = b3.plus( a3 ).minus( b3.componentTimes( a3 ).timesScalar( 2 ) );
          break;
        }
        case RenderBlendType.Hue: {
          c3 = RenderBlendCompose.setLum( RenderBlendCompose.setSat( a3, RenderBlendCompose.sat( b3 ) ), RenderBlendCompose.lum( b3 ) );
          break;
        }
        case RenderBlendType.Saturation: {
          c3 = RenderBlendCompose.setLum( RenderBlendCompose.setSat( b3, RenderBlendCompose.sat( a3 ) ), RenderBlendCompose.lum( b3 ) );
          break;
        }
        case RenderBlendType.Color: {
          c3 = RenderBlendCompose.setLum( a3, RenderBlendCompose.lum( b3 ) );
          break;
        }
        case RenderBlendType.Luminosity: {
          c3 = RenderBlendCompose.setLum( b3, RenderBlendCompose.lum( a3 ) );
          break;
        }
        default: {
          c3 = a3;
          break;
        }
      }

      blended = RenderColor.premultiply( new Vector4( c3.x, c3.y, c3.z, a.w ) );
    }

    let fa;
    let fb;

    // over: fa: 1,   fb: 1-a   fa*a: a      fb*b: b(1-a) sum: a + b(1-a)
    // in:   fa: b,   fb: 0     fa*a: ab     fb*b: 0      sum: ab
    // out:  fa: 1-b, fb: 0     fa*a: a(1-b) fb*b: 0      sum: a(1-b)
    // atop: fa: b,   fb: 1-a   fa*a: ab     fb*b: b(1-a) sum: b
    // xor:  fa: 1-b, fb: 1-a   fa*a: a(1-b) fb*b: b(1-a) sum: a(1-b) + b(1-a)
    // plus: fa: 1,   fb: 1     fa*a: a      fb*b: b      sum: a + b
    switch( composeType ) {
      case RenderComposeType.Over:
        fa = 1;
        fb = 1 - a.w;
        break;
      case RenderComposeType.In:
        fa = b.w;
        fb = 0;
        break;
      case RenderComposeType.Out:
        fa = 1 - b.w;
        fb = 0;
        break;
      case RenderComposeType.Atop:
        fa = b.w;
        fb = 1 - a.w;
        break;
      case RenderComposeType.Xor:
        fa = 1 - b.w;
        fb = 1 - a.w;
        break;
      case RenderComposeType.Plus:
        fa = 1;
        fb = 1;
        break;
      case RenderComposeType.PlusLighter:
        return new Vector4(
          Math.min( a.x + b.x, 1 ),
          Math.min( a.y + b.y, 1 ),
          Math.min( a.z + b.z, 1 ),
          Math.min( a.w + b.w, 1 )
        );
      default:
        throw new Error( 'unimplemented composeType' );
    }

    return new Vector4(
      // Modes like COMPOSE_PLUS can generate alpha > 1.0, so clamp.
      Math.min( fa * blended.x + fb * b.x, 1 ),
      Math.min( fa * blended.y + fb * b.y, 1 ),
      Math.min( fa * blended.z + fb * b.z, 1 ),
      Math.min( fa * a.w + fb * b.w, 1 )
    );
  }

  public static screen( cb: Vector3, cs: Vector3 ): Vector3 {
    return cb.plus( cs ).minus( cb.componentTimes( cs ) );
  }

  public static colorDodge( cb: number, cs: number ): number {
    if ( cb === 0 ) {
      return 0;
    }
    else if ( cs === 1 ) {
      return 1;
    }
    else {
      return Math.min( 1, cb / ( 1 - cs ) );
    }
  }

  public static colorBurn( cb: number, cs: number ): number {
    if ( cb === 1 ) {
      return 1;
    }
    else if ( cs === 0 ) {
      return 0;
    }
    else {
      return 1 - Math.min( 1, ( 1 - cb ) / cs );
    }
  }

  // TODO: check examples, tests
  public static hardLight( cb: Vector3, cs: Vector3 ): Vector3 {
    const falseOption = RenderBlendCompose.screen( cb, cs.timesScalar( 2 ).minusScalar( 1 ) );
    const trueOption = cb.timesScalar( 2 ).componentTimes( cs );

    return new Vector3(
      cs.x <= 0.5 ? trueOption.x : falseOption.x,
      cs.y <= 0.5 ? trueOption.y : falseOption.y,
      cs.z <= 0.5 ? trueOption.z : falseOption.z
    );
  }

  // TODO: check examples, tests
  public static softLight( cb: Vector3, cs: Vector3 ): Vector3 {
    const d = new Vector3(
      cb.x <= 0.25 ? ( 16 * cb.x - 12 ) * cb.x + 4 : Math.sqrt( cb.x ),
      cb.y <= 0.25 ? ( 16 * cb.y - 12 ) * cb.y + 4 : Math.sqrt( cb.y ),
      cb.z <= 0.25 ? ( 16 * cb.z - 12 ) * cb.z + 4 : Math.sqrt( cb.z )
    );
    return new Vector3(
      cs.x > 0.5 ? cb.x + ( 2 * cs.x - 1 ) * ( d.x - cb.x ) : cb.x - ( 1 - 2 * cs.x ) * cb.x * ( 1 - cb.x ),
      cs.y > 0.5 ? cb.y + ( 2 * cs.y - 1 ) * ( d.y - cb.y ) : cb.y - ( 1 - 2 * cs.y ) * cb.y * ( 1 - cb.y ),
      cs.z > 0.5 ? cb.z + ( 2 * cs.z - 1 ) * ( d.z - cb.z ) : cb.z - ( 1 - 2 * cs.z ) * cb.z * ( 1 - cb.z )
    );
  }

  // TODO: check examples, tests
  public static sat( c: Vector3 ): number {
    return Math.max( c.x, Math.max( c.y, c.z ) ) - Math.min( c.x, Math.min( c.y, c.z ) );
  }

  // TODO: check examples, tests
  public static lum( c: Vector3 ): number {
    return 0.3 * c.x + 0.59 * c.y + 0.11 * c.z;
  }

  // TODO: check examples, tests
  public static clipColor( c: Vector3 ): Vector3 {
    const lScalar = RenderBlendCompose.lum( c );
    const l = new Vector3( lScalar, lScalar, lScalar );
    const n = Math.min( c.x, c.y, c.z );
    const x = Math.max( c.x, c.y, c.z );
    if ( n < 0 ) {
      c = l.plus( c.minus( l ).componentTimes( l ).dividedScalar( lScalar - n ) );
    }
    if ( x > 1 ) {
      c = l.plus( c.minus( l ).timesScalar( ( 1 - lScalar ) / ( x - lScalar ) ) );
    }
    return c;
  }

  public static setLum( c: Vector3, l: number ): Vector3 {
    return RenderBlendCompose.clipColor( c.plusScalar( l - RenderBlendCompose.lum( c ) ) );
  }

  public static setSatInner( c: Vector3, s: number ): void {
    if ( c.z > c.x ) {
      c.y = ( ( c.y - c.x ) * s ) / ( c.z - c.x );
      c.z = s;
    }
    else {
      c.y = 0;
      c.z = 0;
    }
    c.x = 0;
  }

  public static setSat( c: Vector3, s: number ): Vector3 {
    // TODO: this is horribly inefficient and wonky

    if ( c.x <= c.y ) {
      if ( c.y <= c.z ) {
        RenderBlendCompose.setSatInner( c, s );
        return c;
      }
      else {
        if ( c.x <= c.z ) {
          const v = new Vector3( c.x, c.z, c.y );
          RenderBlendCompose.setSatInner( v, s );
          return new Vector3( v.x, v.z, v.y );
        }
        else {
          const v = new Vector3( c.z, c.x, c.y );
          RenderBlendCompose.setSatInner( v, s );
          return new Vector3( v.y, v.z, v.x );
        }
      }
    }
    else {
      if ( c.x <= c.z ) {
        const v = new Vector3( c.y, c.x, c.z );
        RenderBlendCompose.setSatInner( v, s );
        return new Vector3( v.y, v.x, v.z );
      }
      else {
        if ( c.y <= c.z ) {
          const v = new Vector3( c.y, c.z, c.x );
          RenderBlendCompose.setSatInner( v, s );
          return new Vector3( v.z, v.x, v.y );
        }
        else {
          const v = new Vector3( c.z, c.y, c.x );
          RenderBlendCompose.setSatInner( v, s );
          return new Vector3( v.z, v.y, v.x );
        }
      }
    }
  }

  public override serialize(): SerializedRenderBlendCompose {
    return {
      type: 'RenderBlendCompose',
      composeType: this.composeType,
      blendType: this.blendType,
      a: this.a.serialize(),
      b: this.b.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderBlendCompose ): RenderBlendCompose {
    return new RenderBlendCompose( obj.composeType, obj.blendType, RenderProgram.deserialize( obj.a ), RenderProgram.deserialize( obj.b ) );
  }
}

scenery.register( 'RenderBlendCompose', RenderBlendCompose );

export type SerializedRenderBlendCompose = {
  type: 'RenderBlendCompose';
  composeType: RenderComposeType;
  blendType: RenderBlendType;
  a: SerializedRenderProgram;
  b: SerializedRenderProgram;
};
